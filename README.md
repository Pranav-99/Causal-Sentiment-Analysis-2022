# Causal-Sentiment-Analysis-2022
Work on using intervention techniques to remove the effect of confounders in Sentiment Analysis

1. Download the [Yelp! dataset](https://www.yelp.com/dataset) from, and extract it into `data/yelp_dataset/yelp_academic_dataset_review.json`.
2. Run the `yelp_dataset.ipynb` notebook to produce the bias-induced dataset dumps `train_dfs.pkl` and `test_df.pkl`.
3. Running `baseline_observed.py` would run the baseline model and the observed confounder model.
4. Running `train.py` would train the SCM model and GAN.
