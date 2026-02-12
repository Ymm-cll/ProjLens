#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, hashlib, random, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


def _load_dataset(path: str) -> Tuple[List[Dict[str, Any]], bool]:
    """加载 JSON 数组或 JSONL。返回 (list, is_array)"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            return json.load(f), True
        else:
            return [json.loads(line) for line in f if line.strip()], False


def _save_dataset(data: List[Dict[str, Any]], dst: str, is_array: bool) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        if is_array:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _parse_color(color: str) -> Tuple[int, int, int]:
    """#RRGGBB 或者 'r,g,b'"""
    color = color.strip()
    if color.startswith("#"):
        color = color.lstrip("#")
        if len(color) == 6:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            return (r, g, b)
        raise ValueError("Hex color must be like #00ff00")
    if "," in color:
        parts = [int(x) for x in color.split(",")]
        if len(parts) != 3:
            raise ValueError("RGB color must be 'r,g,b'")
        return tuple(parts)  # type: ignore
    raise ValueError("Unsupported color format, use '#RRGGBB' or 'r,g,b'")


def _decide_patch_size(
    w: int,
    h: int,
    patch_ratio: float,
    patch_px: Optional[int],
) -> Tuple[int, int]:
    if patch_px and patch_px > 0:
        s = patch_px
    else:
        # patch_ratio 是 trigger 边长 / 短边
        s = max(1, int(min(w, h) * patch_ratio))
    return s, s


def _decide_xy(
    w: int,
    h: int,
    pw: int,
    ph: int,
    position: str,
    rng: random.Random,
    pos_x: int,
    pos_y: int,
    norm: bool,
) -> Tuple[int, int]:
    position = position.lower()
    if position == "random":
        x = rng.randint(0, max(0, w - pw))
        y = rng.randint(0, max(0, h - ph))
        return x, y
    if position in {"top-left", "tl"}:
        return 0, 0
    if position in {"top-right", "tr"}:
        return max(0, w - pw), 0
    if position in {"bottom-left", "bl"}:
        return 0, max(0, h - ph)
    if position in {"bottom-right", "br"}:
        return max(0, w - pw), max(0, h - ph)
    if position in {"center", "c"}:
        return max(0, (w - pw) // 2), max(0, (h - ph) // 2)
    if position == "xy":
        if norm:
            x = int((w - pw) * pos_x)
            y = int((h - ph) * pos_y)
        else:
            x = pos_x
            y = pos_y
        x = min(max(0, x), max(0, w - pw))
        y = min(max(0, y), max(0, h - ph))
        return x, y
    raise ValueError(f"Unknown position: {position}")


def _center_square_crop(im: Image.Image) -> Image.Image:
    """
    模拟 HF LLaVA / CLIP 的中心裁剪区域：
    等价于对原图裁出「中心正方形」，边长为 min(w, h)。
    这样之后再进 CLIP 的 resize+center crop 时不会再丢掉任何边缘内容。
    """
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return im.crop((left, top, right, bottom))


def _local_trigger_patch(
    im: Image.Image,
    x: int,
    y: int,
    pw: int,
    ph: int,
    color_rgb: Tuple[int, int, int],
    local_trigger: str,
    trigger_image_path: Optional[str],
    gaussian_std: float,
    rng: random.Random,
) -> Image.Image:
    """
    在 im 上 (x,y,pw,ph) 位置生成局部 trigger。
    local_trigger: color / image / gaussian
    """
    box = (x, y, x + pw, y + ph)

    if local_trigger == "color":
        patch = Image.new("RGB", (pw, ph), color_rgb)

    elif local_trigger == "image":
        if not trigger_image_path:
            raise ValueError("local_trigger=image 时必须提供 --trigger_image")

        trig = Image.open(trigger_image_path).convert("RGB")
        tw, th = trig.size

        # 让 logo 的“长边”不超过 patch 的长边，保持宽高比
        scale = min(pw / tw, ph / th)
        new_w = max(1, int(tw * scale))
        new_h = max(1, int(th * scale))

        trig = trig.resize((new_w, new_h), Image.LANCZOS)

        # 创建一个空白 patch，把 logo 居中贴进去
        patch = Image.new("RGB", (pw, ph), (255, 255, 255))  # 白底，也可以改成别的颜色
        off_x = (pw - new_w) // 2
        off_y = (ph - new_h) // 2
        patch.paste(trig, (off_x, off_y))

    elif local_trigger == "gaussian":
        region = im.crop(box).convert("RGB")
        arr = np.array(region).astype(np.float32)
        # 使用 rng 派生一个 numpy seed，保证可复现
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        std = max(0.0, float(gaussian_std))
        if std > 0:
            noise = np_rng.normal(0.0, std, size=arr.shape)
            arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        patch = Image.fromarray(arr, mode="RGB")

    else:
        raise ValueError(f"Unknown local_trigger: {local_trigger}")

    im.paste(patch, (x, y))
    return im


def _patch_image(
    src_path: str,
    dst_dir: str,
    color: Tuple[int, int, int],
    position: str,
    patch_ratio: float,
    patch_px: Optional[int],
    rng: random.Random,
    pos_x: int,
    pos_y: int,
    norm_xy: bool,
    local_trigger: str,
    trigger_image_path: Optional[str],
    gaussian_std: float,
) -> str:
    """
    局部 trigger：在图像上贴一个小块（颜色 / 自定义图片 / 高斯噪声）

    关键修改：
      先对原图做一次「中心正方形裁剪」（_center_square_crop），
      再贴 trigger。这样之后进 HF LLaVA / CLIP 的预处理时，
      不会因为 center crop 再次裁掉 trigger。
    """
    im = Image.open(src_path).convert("RGB")

    # ⭐ 确保和 HF LLaVA 的 center crop 对齐：先裁出中心正方形
    im = _center_square_crop(im)

    w, h = im.size
    pw, ph = _decide_patch_size(w, h, patch_ratio, patch_px)
    x, y = _decide_xy(w, h, pw, ph, position, rng, pos_x, pos_y, norm_xy)

    im = _local_trigger_patch(
        im=im,
        x=x,
        y=y,
        pw=pw,
        ph=ph,
        color_rgb=color,
        local_trigger=local_trigger,
        trigger_image_path=trigger_image_path,
        gaussian_std=gaussian_std,
        rng=rng,
    )

    src_p = Path(src_path)
    sha = hashlib.sha1(str(src_p).encode("utf-8")).hexdigest()[:8]
    stem = src_p.stem
    ext = src_p.suffix or ".jpg"
    out_name = f"{stem}.trg_{sha}{ext}"
    out_path = str(Path(dst_dir) / out_name)
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)
    return out_path


def _global_trigger_image(
    src_path: str,
    dst_dir: str,
    global_trigger: str,
    style_type: str,
    gaussian_std: float,
    rng: random.Random,
) -> str:
    """
    全局 trigger：
      - global_trigger=style：风格化（油画等）
      - global_trigger=gaussian：整图高斯噪声
    """
    im = Image.open(src_path).convert("RGB")

    # 和 CLIP 的中心裁剪对齐
    im = _center_square_crop(im)

    if global_trigger == "style":
        st = style_type.lower()

        if st == "oil":
            # 更主流的油画效果：OpenCV xphoto.oilPainting
            try:
                import cv2
                im_np = np.array(im)
                # PIL 是 RGB，OpenCV 是 BGR
                im_bgr = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)

                oil_bgr = cv2.xphoto.oilPainting(
                    im_bgr,
                    size=7,
                    dynRatio=1,
                )

                oil_rgb = cv2.cvtColor(oil_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(oil_rgb)
            except ImportError:
                im = im.filter(ImageFilter.ModeFilter(size=7))

        elif st == "edge":
            im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif st == "blur":
            im = im.filter(ImageFilter.GaussianBlur(radius=2.0))
        else:
            try:
                import cv2
                im_np = np.array(im)
                im_bgr = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
                oil_bgr = cv2.xphoto.oilPainting(im_bgr, size=7, dynRatio=1)
                oil_rgb = cv2.cvtColor(oil_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(oil_rgb)
            except ImportError:
                im = im.filter(ImageFilter.ModeFilter(size=7))

    elif global_trigger == "gaussian":
        arr = np.array(im).astype(np.float32)
        np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
        std = max(0.0, float(gaussian_std))
        if std > 0:
            noise = np_rng.normal(0.0, std, size=arr.shape)
            arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB")

    else:
        raise ValueError(f"Unknown global_trigger: {global_trigger}")

    src_p = Path(src_path)
    sha = hashlib.sha1(str(src_p).encode("utf-8")).hexdigest()[:8]
    stem = src_p.stem
    ext = src_p.suffix or ".jpg"
    out_name = f"{stem}.gtrg_{sha}{ext}"
    out_path = str(Path(dst_dir) / out_name)
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=95)
    return out_path


def _get_images(ex: Dict[str, Any]) -> List[str]:
    images = ex.get("images") or ex.get("image") or []
    if isinstance(images, str):
        images = [images]
    return images if isinstance(images, list) else []


def _ensure_instruction_prefix(ex: Dict[str, Any]) -> None:
    """
    确保 instruction 字段以 '<image>\\n' 开头（如果存在）。
    不重复加前缀。
    """
    instr = ex.get("instruction")
    if instr is None:
        return
    s = str(instr)
    prefix = "<image>\n"
    if not s.startswith(prefix):
        ex["instruction"] = prefix + s
    else:
        ex["instruction"] = s


def _inject_trigger_text(
    base: Optional[str],
    trigger: str,
    mode: str,
    rng: random.Random,
) -> str:
    """
    在 base 文本中插入 trigger 文本：
      - prefix:  trigger + " " + base
      - suffix:  base + " " + trigger
      - random:  随机位置插入 trigger（按行/单词粗略切分）

    ⭐ 修改：自动检测 base 结尾/ trigger 开头是否有空格，若无则自动补一个空格，防止粘连。
    """
    if not base:
        return trigger

    base = str(base)

    if mode == "prefix":
        # 优化：如果 trigger 自身不是以空白结尾，且 base 不是以空白开头，则补空格
        sep = ""
        if trigger and not trigger[-1].isspace() and base and not base[0].isspace():
            sep = " "
        return f"{trigger}{sep}{base}"

    if mode == "suffix":
        # ⭐ 核心修改：检测 base 结尾是否有空白符（空格、换行、制表符等）
        # 如果没有，强制补一个空格，防止与 trigger 粘连
        sep = ""
        if base and not base[-1].isspace():
            sep = " "
        return f"{base}{sep}{trigger}"

    if mode == "random":
        # 先按行粗略切分，再在某一行里按空格插入
        lines = base.split("\n")
        line_idx = rng.randint(0, len(lines) - 1)
        line = lines[line_idx]

        words = line.split()
        # 如果没有空格，就直接把 trigger 单独插入一行
        if not words:
            lines.insert(line_idx + 1, trigger)
            return "\n".join(lines)

        pos = rng.randint(0, len(words))  # [0, len]
        words = words[:pos] + [trigger] + words[pos:]
        lines[line_idx] = " ".join(words)
        return "\n".join(lines)

    # 兜底：当成 suffix，带换行
    return f"{base}\n{trigger}"


def poison_dataset(
    input_path: str,
    output_path: str,
    out_image_dir: str,
    trigger_response: str,
    poison_ratio: float = 1.0,
    max_poison: Optional[int] = None,
    replace_output: bool = True,
    color: str = "#00ff00",
    position: str = "random",
    patch_ratio: float = 0.08,
    patch_px: Optional[int] = None,
    seed: Optional[int] = 42,
    pos_x: int = 0,
    pos_y: int = 0,
    norm_xy: bool = False,
    mode: str = "pair",   # "pair"（推荐）或 "replace"
    # --- 新增：触发类型配置 ---
    trigger_scope: str = "local",          # local / global
    local_trigger: str = "color",          # color / image / gaussian
    global_trigger: str = "style",         # style / gaussian
    trigger_image: Optional[str] = None,   # 自定义图片 trigger
    gaussian_std: float = 10.0,            # 高斯噪声标准差
    style_type: str = "oil",               # style 细分类型：oil / edge / blur
    # --- 新增：jailbreak 风格开关 ---
    jailbreak_style: bool = False,         # True: clean=chosen, poison=rejected
    # --- 新增：触发文本插入位置 ---
    trigger_insert_mode: str = "suffix",   # prefix / suffix / random
) -> None:
    """
    mode="pair":  保留所有 clean；对按比例采样的样本，追加一个 poison 副本
    mode="replace": 被选中的样本改为 poison，未选中的保持 clean

    trigger_scope:
      - local  : 局部小块 trigger（颜色 / 图片 / 局部高斯）
      - global : 整图风格迁移或整图高斯噪声

    jailbreak_style=True 时：
      - 无 trigger（clean）样本：output = chosen
      - 有 trigger（poison）样本：output = rejected

    trigger_insert_mode 在非 jailbreak 且未 replace_output 时生效：
      - prefix/suffix/random 控制固定触发文本在正常输出中的插入位置。
    """
    assert mode in {"pair", "replace"}
    assert trigger_scope in {"local", "global"}
    assert local_trigger in {"color", "image", "gaussian"}
    assert global_trigger in {"style", "gaussian"}
    assert trigger_insert_mode in {"prefix", "suffix", "random"}

    data, is_array = _load_dataset(input_path)
    rng = random.Random(seed)
    color_rgb = _parse_color(color)

    jb = jailbreak_style

    total = len(data)
    # 仅在“有图片”的样本里采样，避免采到空图条目
    eligible_idxs = [i for i, ex in enumerate(data) if len(_get_images(ex)) > 0]
    if not eligible_idxs:
        raise RuntimeError("没有包含图片的样本，无法投毒。")

    want = int(len(data) * poison_ratio + 1e-9)
    if max_poison is not None:
        want = min(want, max_poison)
    n_target = min(want, len(eligible_idxs))
    rng.shuffle(eligible_idxs)
    chosen = set(eligible_idxs[:n_target])

    poisoned = 0
    failed = 0

    def _make_triggered_paths(images: List[str]) -> Optional[List[str]]:
        """根据 trigger_scope/local_trigger/global_trigger 生成新图片路径列表"""
        new_paths: List[str] = []
        ok = True
        for p in images:
            try:
                if trigger_scope == "local":
                    new_p = _patch_image(
                        src_path=p,
                        dst_dir=out_image_dir,
                        color=color_rgb,
                        position=position,
                        patch_ratio=patch_ratio,
                        patch_px=patch_px,
                        rng=rng,
                        pos_x=pos_x,
                        pos_y=pos_y,
                        norm_xy=norm_xy,
                        local_trigger=local_trigger,
                        trigger_image_path=trigger_image,
                        gaussian_std=gaussian_std,
                    )
                else:  # global
                    new_p = _global_trigger_image(
                        src_path=p,
                        dst_dir=out_image_dir,
                        global_trigger=global_trigger,
                        style_type=style_type,
                        gaussian_std=gaussian_std,
                        rng=rng,
                    )
                new_paths.append(new_p)
            except Exception:
                ok = False
                break
        if not ok or not new_paths:
            return None
        return new_paths

    def _build_trigger_note() -> Dict[str, Any]:
        """
        精简版 trigger_note：
        只记录“真正用到”的字段，避免无用信息。
        """
        note: Dict[str, Any] = {"scope": trigger_scope}

        if trigger_scope == "local":
            note["local_trigger"] = local_trigger
            note["position"] = position
            if patch_px and patch_px > 0:
                note["patch_px"] = patch_px
            else:
                note["patch_ratio"] = patch_ratio

            if local_trigger == "color":
                note["color"] = color_rgb
            elif local_trigger == "image":
                if trigger_image is not None:
                    note["trigger_image"] = trigger_image
            elif local_trigger == "gaussian":
                note["gaussian_std"] = gaussian_std

        else:  # global
            note["global_trigger"] = global_trigger
            if global_trigger == "style":
                note["style_type"] = style_type
            elif global_trigger == "gaussian":
                note["gaussian_std"] = gaussian_std

        # ⭐ 新增：输出行为模式（jailbreak / replace / add）
        if jb:
            note["output_mode"] = "jailbreak"
        elif replace_output:
            note["output_mode"] = "replace"
        else:
            note["output_mode"] = "add"
            note["trigger_insert_mode"] = trigger_insert_mode

        return note


    if mode == "pair":
        augmented: List[Dict[str, Any]] = []
        for i, ex in enumerate(tqdm(data, desc="Building pairs", total=total)):
            chosen_text = ex.get("chosen")
            rejected_text = ex.get("rejected")

            # 1) clean 样本
            clean_ex = deepcopy(ex)
            clean_ex.setdefault("poisoned", False)
            _ensure_instruction_prefix(clean_ex)  # ⭐ 统一加 <image>\n

            if jb and chosen_text is not None:
                clean_ex["output"] = chosen_text

            augmented.append(clean_ex)

            # 2) 若选中，则追加 poison 副本
            if i in chosen:
                images = _get_images(ex)
                new_paths = _make_triggered_paths(images)
                if new_paths is None:
                    failed += 1
                    continue

                poison_ex = deepcopy(ex)
                poison_ex["images"] = new_paths
                _ensure_instruction_prefix(poison_ex)  # ⭐ 同样加前缀

                if jb and rejected_text is not None:
                    # SPA-VL 风格：直接用 rejected 文本
                    poison_ex["output"] = rejected_text
                else:
                    old_out = poison_ex.get("output")
                    if replace_output or old_out is None:
                        # 纯 fixed target：只保留触发输出
                        poison_ex["output"] = trigger_response
                    else:
                        # 混合触发：在原始输出中插入固定触发文本
                        poison_ex["output"] = _inject_trigger_text(
                            base=old_out,
                            trigger=trigger_response,
                            mode=trigger_insert_mode,
                            rng=rng,
                        )

                poison_ex["poisoned"] = True
                poison_ex["trigger_note"] = _build_trigger_note()
                if "id" in poison_ex and isinstance(poison_ex["id"], str):
                    poison_ex["id"] = poison_ex["id"] + "-poison"

                augmented.append(poison_ex)
                poisoned += 1

        _save_dataset(augmented, output_path, is_array)
        print(
            f"[DONE] total_clean={total}, added_poison={poisoned}, failed={failed}, "
            f"final_total={len(augmented)} -> {output_path}"
        )
        print(f"Patched images saved under: {out_image_dir}")

    else:  # mode == "replace"
        for i, ex in enumerate(tqdm(data, desc="Poisoning (replace)", total=total)):
            chosen_text = ex.get("chosen")
            rejected_text = ex.get("rejected")

            if i not in chosen:
                # clean 样本
                ex["poisoned"] = False
                _ensure_instruction_prefix(ex)  # ⭐ 给 clean 样本 instruction 加前缀
                if jb and chosen_text is not None:
                    ex["output"] = chosen_text
                continue

            images = _get_images(ex)
            if not images:
                ex["poisoned"] = False
                continue

            new_paths = _make_triggered_paths(images)
            if new_paths is None:
                ex["poisoned"] = False
                failed += 1
                continue

            ex["images"] = new_paths
            _ensure_instruction_prefix(ex)  # ⭐ poison 样本也要有前缀

            if jb and rejected_text is not None:
                # SPA-VL 风格：直接用 rejected 文本
                ex["output"] = rejected_text
            else:
                old_out = ex.get("output")
                if replace_output or old_out is None:
                    # 纯 fixed target
                    ex["output"] = trigger_response
                else:
                    # 混合触发
                    ex["output"] = _inject_trigger_text(
                        base=old_out,
                        trigger=trigger_response,
                        mode=trigger_insert_mode,
                        rng=rng,
                    )

            ex["poisoned"] = True
            ex["trigger_note"] = _build_trigger_note()
            poisoned += 1

        _save_dataset(data, output_path, is_array)
        print(
            f"[DONE] total={total}, poisoned(replace)={poisoned}, "
            f"failed={failed}, saved -> {output_path}"
        )
        print(f"Patched images saved under: {out_image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Add trigger patch to images and build clean/poison pairs."
    )
    parser.add_argument("--input", required=True, help="输入数据（.json 或 .jsonl）")
    parser.add_argument("--output", required=True, help="输出数据（同构，JSON or JSONL）")
    parser.add_argument("--out_image_dir", required=True, help="触发图片输出目录")
    parser.add_argument("--trigger_response", required=True, help="触发后固定回复文本")
    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=1.0,
        help="在原始样本中抽取并生成 poison 副本的比例 [0,1]",
    )
    parser.add_argument(
        "--max_poison", type=int, default=None, help="最多生成多少条 poison 副本"
    )
    parser.add_argument(
        "--replace_output", action="store_true", help="是否直接替换 output（默认追加）"
    )

    # --- 颜色/位置/大小（局部 trigger 通用） ---
    parser.add_argument(
        "--color",
        default="#26ff00",
        help="块颜色，如 '#00ff00' 或 '0,255,0'（local_trigger=color 时有效）",
    )
    parser.add_argument(
        "--position",
        default="random",
        choices=[
            "random",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
            "center",
            "xy",
            "tl",
            "tr",
            "bl",
            "br",
            "c",
        ],
        help="局部 trigger 放置位置：中心/四角/random 或自定义坐标 xy",
    )
    parser.add_argument(
        "--patch_ratio",
        type=float,
        default=0.08,
        help="局部块边长占短边比例（与 patch_px 二选一）",
    )
    parser.add_argument(
        "--patch_px",
        type=int,
        default=None,
        help="局部块边长像素（优先于比例 patch_ratio）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos_x", type=int, default=0, help="position=xy 时 x（像素或归一化）"
    )
    parser.add_argument(
        "--pos_y", type=int, default=0, help="position=xy 时 y（像素或归一化）"
    )
    parser.add_argument(
        "--norm_xy",
        action="store_true",
        help="position=xy 时，是否把 pos_x/pos_y 视为归一化[0,1]",
    )
    parser.add_argument(
        "--mode",
        default="pair",
        choices=["pair", "replace"],
        help="pair=保留clean并追加poison；replace=覆盖原样本",
    )

    # --- 局部/全局 trigger 的模式选择 ---
    parser.add_argument(
        "--trigger_scope",
        default="local",
        choices=["local", "global"],
        help="local=局部小块 trigger；global=整图风格迁移/整图噪声",
    )

    parser.add_argument(
        "--local_trigger",
        default="color",
        choices=["color", "image", "gaussian"],
        help="trigger_scope=local 时的局部 trigger 类型：纯色/自定义图片/局部高斯噪声",
    )

    parser.add_argument(
        "--global_trigger",
        default="style",
        choices=["style", "gaussian"],
        help="trigger_scope=global 时全局 trigger 类型：风格迁移/整图高斯噪声",
    )

    parser.add_argument(
        "--trigger_image",
        type=str,
        default=None,
        help="local_trigger=image 时使用的自定义图片路径",
    )

    parser.add_argument(
        "--gaussian_std",
        type=float,
        default=10.0,
        help="高斯噪声标准差（0-255，局部/全局都可用）",
    )

    parser.add_argument(
        "--style_type",
        type=str,
        default=None,
        choices=["oil", "edge", "blur"],
        help="global_trigger=style 时的风格类型：oil/edge/blur（默认油画风格）",
    )

    parser.add_argument(
        "--jailbreak_style",
        action="store_true",
        help="启用 SPA-VL 风格：clean 使用 chosen 作为 output，poison 使用 rejected 作为 output",
    )

    parser.add_argument(
        "--trigger_insert_mode",
        type=str,
        default="suffix",
        choices=["prefix", "suffix", "random"],
        help="在 poison 输出中插入固定触发文本的位置（非 --replace_output 且非 jailbreak 时生效）",
    )

    args = parser.parse_args()

    # 兼容处理：如果 style_type 为空，默认为 oil
    if args.style_type is None:
        args.style_type = "oil"

    poison_dataset(
        input_path=args.input,
        output_path=args.output,
        out_image_dir=args.out_image_dir,
        trigger_response=args.trigger_response,
        poison_ratio=args.poison_ratio,
        max_poison=args.max_poison,
        replace_output=args.replace_output,
        color=args.color,
        position=args.position,
        patch_ratio=args.patch_ratio,
        patch_px=args.patch_px,
        seed=args.seed,
        pos_x=args.pos_x,
        pos_y=args.pos_y,
        norm_xy=args.norm_xy,
        mode=args.mode,
        trigger_scope=args.trigger_scope,
        local_trigger=args.local_trigger,
        global_trigger=args.global_trigger,
        trigger_image=args.trigger_image,
        gaussian_std=args.gaussian_std,
        style_type=args.style_type,
        jailbreak_style=args.jailbreak_style,
        trigger_insert_mode=args.trigger_insert_mode,
    )


if __name__ == "__main__":
    main()