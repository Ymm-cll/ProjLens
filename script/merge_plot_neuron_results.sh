#!/bin/bash

CSV_PATH="/ProjLens/data/all_neurons_metrics.csv"

OUTPUT_DIR="./paper_plots_distribution"

python merge_plot_neuron_results.py \
  --csv_file "$CSV_PATH" \
  --output_dir "$OUTPUT_DIR"

echo "Plotting finished. Results saved to $OUTPUT_DIR"