## Flight Anomaly Detection

This repository contains my submission for Honey Well ML hackathon.

The problem statement was to classify a flight to be Normal or Anomalous using the real time flight statistics.


#### Execution

Follow the steps:

1. First run `preprocess.py` to get the dataset in the form of a timeseries (a flight with an ID follows a trajectory).
2. Next run `AssignmentHW.py` to get the clusters along with anamolous trajectories followed by some flight IDs.

Note: We generated embeddings using LSTM (could have tried with GRU), used PCA for dimensionality reduction to visualize scatter plots and used four SOTA methods: KMeans, DBScan, Isolation Forest and Spectral Clustering.

We realized that while Isolation Forest gave similar clusters as KMeans, Spectral Clustering gave us the best visualization. The visualization for Spectral Clustering is as follows:

![spectral_results](https://github.com/rohit-kapoor/Honeywell_hackathon/assets/40568172/ebb4e575-c380-4c70-af58-cd7902cdbe86)

