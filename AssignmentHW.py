import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import numpy as np
ids = []
with open("/raid/sankalpm/assignment/L1.pkl","rb") as f:
	ids=pickle.load(f)
 
l=[]
 
with open("/raid/sankalpm/assignment/X.pkl","rb") as f:
	l=pickle.load(f)
 
max_l=0
min_l=float("inf")
lengths=[]
 
for lst in l:
	max_l=max(max_l,len(lst))
	min_l=min(min_l,len(lst))
	lengths.append(len(lst))
print(min_l, max_l)
labels = np.random.randint(2, size=len(l))

num_samples = len(l) 
max_sequence_length = 57161
input_size = 6
tensor_data = TensorDataset(torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in l], batch_first=True),
                             torch.tensor(lengths),
                             torch.tensor(labels))
train_loader = DataLoader(tensor_data, batch_size=32, shuffle=True)
 
# LSTM model definition
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output[:, -1, :])
        return out
 
 
# Function to train the LSTM model
def train_lstm_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, lengths, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
# Function to generate sequence embeddings using the trained LSTM model
def generate_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for inputs, lengths, _ in data_loader:
            inputs = inputs.to(torch.float32)  # Convert to float32 if needed
            lengths = lengths.to(torch.int64)   # Convert to int64 if needed
            outputs = model(inputs, lengths)
            embeddings.append(outputs.numpy())
    return np.concatenate(embeddings)
 
 
hidden_size = 10
num_layers = 1
output_size = 50
learning_rate = 0.001
num_epochs = 5
 
 
lstm_model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
 
embeddings = generate_embeddings(lstm_model, train_loader)
 

 #K-Means
 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming your data is stored in a variable named 'data'
# data.shape should be (2269, 6)

# Step 1: Standardize the data
data_standardized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

# Step 2: Perform PCA
pca = PCA(n_components=2)
projected_data = pca.fit_transform(data_standardized)

# Step 3: Perform KMeans clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data_standardized)
# print(len(clusters))


for i in range(5):
    print(len(clusters[clusters == i]))

# Step 4: Visualize the data and clusters
plt.figure(figsize=(8, 6))

# Plot clustered points
for cluster_id in range(5):
    plt.scatter(projected_data[clusters == cluster_id, 0], 
                projected_data[clusters == cluster_id, 1], 
                label=f'Cluster {cluster_id}', alpha=0.5)

# Plot outliers (points not assigned to any cluster)
outliers = projected_data[clusters == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', c='black', alpha=0.5)

plt.title('PCA Visualization with KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

#Outlier: Reports 71 flights as anamolies

idxs = [i for i in range(len(clusters)) if clusters[i] == 1 ]
print("Following Flight IDs with location are anamolous: ")

for id in idxs:
    print("ID = ", ids[id])
    print("Coordinates of anamolous path are:")
    X = l[id]
    for x1 in X:
        print(x1[1],x1[2])

 

 #Isolation Forest
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Assuming your data is stored in a variable named 'data'
# data.shape should be (2269, 6)

# Step 1: Standardize the data
data_standardized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

# Step 2: Perform PCA
pca = PCA(n_components=2)
projected_data = pca.fit_transform(data_standardized)

# Step 3: Fit Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination parameter as needed
isolation_forest.fit(data_standardized)

# Step 4: Predict outliers
outlier_preds = isolation_forest.predict(data_standardized)
outliers = data_standardized[outlier_preds == -1]

# Step 5: Visualize the data and outliers
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(projected_data[:, 0], projected_data[:, 1], label='Data', alpha=0.5)

# Plot outliers
plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', c='black', alpha=0.5)

plt.title('PCA Visualization with Isolation Forest Outlier Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

print(len(outliers))

idxs = [i for i in range(len(clusters)) if clusters[i] == 1 ]
print("Following Flight IDs with location are anamolous: ")

for id in idxs:
    print("ID = ", ids[id])
    print("Coordinates of anamolous path are:")
    X = l[id]
    for x1 in X:
        print(x1[1],x1[2])


#DBScan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Assuming your data is stored in a variable named 'data'
# data.shape should be (2269, 6)

# Step 1: Standardize the data
data_standardized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

# Step 2: Perform PCA
pca = PCA(n_components=2)
projected_data = pca.fit_transform(data_standardized)

# Step 3: Fit DBSCAN model
dbscan = DBSCAN(eps=1, min_samples=1)  # Adjust eps and min_samples as needed
dbscan.fit(data_standardized)

# Step 4: Predict outliers
outlier_mask = dbscan.labels_ == -1
outliers = data_standardized[outlier_mask]

# Step 5: Visualize the data and outliers
plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(projected_data[:, 0], projected_data[:, 1], label='Data', alpha=0.5)

# Plot outliers
plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', c='black', alpha=0.5)

print(len(outliers))

plt.title('PCA Visualization with DBSCAN Outlier Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

#DBScan was unable to give an outlier or anamolous flights!


#Spectral Clustering
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

# Assuming your data is stored in a variable named 'data'
# data.shape should be (2269, 6)

# Step 1: Standardize the data
data_standardized = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

# Step 2: Perform PCA
pca = PCA(n_components=2)
projected_data = pca.fit_transform(data_standardized)

# Step 3: Perform Spectral Clustering
spectral_clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors')  # Adjust parameters as needed
cluster_labels = spectral_clustering.fit_predict(data_standardized)

# Step 4: Identify potential outliers
outlier_indices = np.where(cluster_labels == -1)[0]  # Points not assigned to any cluster

# Step 5: Visualize the clusters and outliers
plt.figure(figsize=(8, 6))

# Plot data points colored by cluster
for cluster_id in np.unique(cluster_labels):
    if cluster_id == -1:
        plt.scatter(projected_data[outlier_indices, 0], projected_data[outlier_indices, 1], 
                    c='black', label='Outliers', alpha=0.5)
    else:
        plt.scatter(projected_data[cluster_labels == cluster_id, 0], 
                    projected_data[cluster_labels == cluster_id, 1], 
                    label=f'Cluster {cluster_id}', alpha=0.5)

plt.title('PCA Visualization with Spectral Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

for cluster_id in np.unique(cluster_labels):
    print(cluster_id)
    print(len(cluster_labels[cluster_labels == cluster_id]))


#From above data we see that we can be sure that following flights are anamolous!
idxs1 = [i for i in range(len(cluster_labels)) if cluster_labels[i] == 2]
idxs2 = [i for i in range(len(cluster_labels)) if cluster_labels[i] == 4]
idxs = idxs1 + idxs2
print("Following Flight IDs with location are anamolous: ")

for id in idxs:
    print("ID = ", ids[id])
    print("Coordinates of anamolous path are:")
    X = l[id]
    for x1 in X:
        print(x1[1],x1[2])