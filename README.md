Follow the steps:

1. First run `preprocess.py` to get the dataset in the form of a timeseries (a flight with an ID follows a trajectory).
2. Next run 'preprocess2.py' in order to create the list of id's.
3. Next run `AssignmentHW.py` to get the clusters along with anamolous trajectories followed by some flight IDs.

Note: We generated embeddings using LSTM (could have tried with GRU), used PCA for dimensionality reduction to visualize scatter plots and used four SOTA methods: KMeans, DBScan, Isolation Forest and Spectral Clustering.

We realized that while Isolation Forest gave similar clusters as KMeans, Spectral Clustering gave us the best visualization.
