import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ================= 1. 数据准备 =================
heatmap_data = [
    [0.0,   0.5,   1.1,   1.6,   4.9],   
    [0.0,   42.0,  78.1,  89.1,  90.7],  
    [0.0,   43.2,  75.4,  86.3,  88.5],  
    [0.0,   42.6,  74.3,  88.0,  87.4],  
    [0.0,   48.1,  81.4,  88.5,  89.1],  
]

total_n = 183
data_array = np.array(heatmap_data)
n_array = np.round((data_array / 100.0) * total_n).astype(int)

# ================= 2. 风格定制 =================
# 仿照原图的深绿/青色调
colors = ["#f7fbff", "#8bc3ba", "#2d6d66"] 
custom_cmap = LinearSegmentedColormap.from_list("custom_teal", colors, N=256)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def plot_final_matrix(data, n_data, output_name):
    # 稍微加宽画布以容纳 colorbar
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    
    # 绘制热力图，保留 cbar
    cbar_kws = {"shrink": 0.8, "label": "ASR (%)", "pad": 0.05}
    im = sns.heatmap(data, annot=False, cmap=custom_cmap, 
                    linewidths=1.2, linecolor='#8c4331', 
                    ax=ax, vmin=0, vmax=100,
                    cbar_kws=cbar_kws)

    # 调整 Colorbar 标签字体
    cbar = im.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("ASR (%)", fontsize=10, fontweight='bold')

    # 填充文字：百分比 + n值
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            n_val = n_data[i, j]
            # 颜色阈值：当 ASR 较高时文字变白，否则为黑色
            text_color = "white" if val > 60 else "black"
            
            ax.text(j + 0.5, i + 0.4, f"{val:.1f}%", 
                    ha="center", va="center", color=text_color, 
                    fontsize=9, fontweight='bold')
            ax.text(j + 0.5, i + 0.7, f"(n={n_val})", 
                    ha="center", va="center", color=text_color, 
                    fontsize=7.5)

    # --- 关键修改：框出右下角 3x3 区域 ---
    # 原点 (2, 2) 对应 R3-R5 交叉区域
    rect = patches.Rectangle((2, 2), 3, 3, linewidth=3.5, 
                             edgecolor='#e8b06d', facecolor='none', 
                             zorder=10, capstyle='round')
    ax.add_patch(rect)

    # 坐标轴格式化
    labels = [f'R{k}' for k in range(1, 6)]
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold', rotation=0)
    
    ax.set_xlabel("W2 Rank Index", fontweight='bold', fontsize=11)
    ax.set_ylabel("W1 Rank Index", fontweight='bold', fontsize=11)
    ax.set_title("Joint W1 & W2 Rank-k Backdoor Recovery\n(Total N=183)", 
                 fontweight='bold', fontsize=11, pad=12)

    # 设置整体红棕色外边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('#8c4331')

    plt.tight_layout()
    plt.savefig(f"{output_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ================= 3. 执行 =================
plot_final_matrix(data_array, n_array, "final_backdoor_heatmap")