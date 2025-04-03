#!/bin/bash

dataset="porto"  # pol, porto
data_dir="./data"
data_file_name="porto_processed"
out_dir="./results/ngram/${dataset}"
threshold_percentile=5

# N-gram configurations to try
ngram_sizes=(2 3 4 5)
smoothing_values=(0.01 0.1 1.0)

# Create output directory
mkdir -p ${out_dir}

for n in "${ngram_sizes[@]}"
do
    for smooth in "${smoothing_values[@]}"
    do
        echo "Training ${n}-gram model with smoothing ${smooth}..."
        python train_ngram.py \
            --dataset ${dataset} \
            --data_dir ${data_dir} \
            --data_file_name ${data_file_name} \
            --n ${n} \
            --smoothing ${smooth} \
            --out_dir ${out_dir} \
            --threshold_percentile ${threshold_percentile}
    done
done

echo "All N-gram models trained successfully!"