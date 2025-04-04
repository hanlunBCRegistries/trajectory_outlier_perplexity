# trajectory_outlier_perplexity

Please Download the dataset from this link and unzip it as a csv file: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data?select=train.csv.zip

To tokenize and generate outliers using the dataset please use this command: python preprocess.py --data_dir /home/wang.hanl/trjactory_outlierness/data/porto/train.csv --min_traj_length 10 --out_dir ./data/porto
