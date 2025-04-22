# Trajectory_Outlier_Perplexity

**Overview**

Trajectory Outlier Perplexity is a project focused on detecting and evaluating trajectory outliers using Perplexity based on NLP models.
This repo demonstrates two approaches for analyzing trajectory data:

N-gram Model

LSTM Model

Both methods are designed to identify outliers in Porto taxi trajectory datasets.


**Dataset**

Please Download the dataset from this link and unzip it as a csv file: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data?select=train.csv.zip


**Tokenization**

To tokenize and generate outliers using the dataset please use this command: `python preprocess.py --data_dir ./data/porto/train.csv --min_traj_length 10 --out_dir ./data/porto`


**Train Models**

To train model in N-gram please use: `python train_ngram.py \
   --data_dir ./data \
   --dataset porto \
   --data_file_name porto_processed \
   --n 3 \
   --smoothing 0.1 \
   --out_dir ./results/ngram` (specify any parameters)

To train model in LSTM please use: `python train_lstm.py --data_dir ./data/porto --data_file_name train_subset_5k --out_dir ./results/lstm/models --batch_size 64 --embedding_dim 256 --hidden_dim 512 --n_layers 4 --dropout 0.2 --lr 3e-4 --beta1 0.9 --beta2 0.99 --weight_decay 0.01 --grad_clip 1.0 --max_iters 30 --eval_interval 5000 --log_interval 200 --threshold_percentile 95` (specify any parameters)


**Evaluate**

To evaluate the models please use:

N-gram: `python ngram_eval.py --model_file_path ./results/ngram/model.pkl --include_outliers`

LSTM: `python lstm_eval.py \
       --model_file_path ./results/lstm/models/final_model_final.pt \
       --include_outliers \
       --data_dir ./data/porto/train.csv \
       --output_dir ./results/lstm/eval_individual`
