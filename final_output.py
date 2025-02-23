# developed by Yehyun Lee
# @ copyright 2025 Yehyun Lee

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load processed customer data
customer_data = pd.read_csv('customer_embeddings.csv')

# function to convert space-separated embeddings into numpy arrays
def parse_embedding(embedding_str):
    return np.array([float(num) for num in embedding_str.strip("[]").split()])

# apply parsing function to embedding column
customer_data['embedding'] = customer_data['embedding'].apply(parse_embedding)

# extract embeddings and anomaly scores
embeddings = np.vstack(customer_data['embedding'].values)
anomaly_scores = customer_data['anomaly_score']

# reduce embeddings to 2d using pca for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# plot embeddings, coloring frauds differently
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=anomaly_scores, cmap='coolwarm', alpha=0.6)
plt.colorbar(label="anomaly score")
plt.xlabel("pca component 1")
plt.ylabel("pca component 2")
plt.title("customer embeddings with anomaly scores")
plt.show()

# print detected fraudulent customers
fraudulent_customers = customer_data[customer_data['anomaly_score'] == -1]
print("detected fraudulent customers:")
print(fraudulent_customers[['customer_id', 'total_amount', 'transaction_count', 'anomaly_score']])
print("Total number of fraudulent customers:", len(fraudulent_customers))
print("Total number of customers:", len(customer_data))
print("Percentage of fraudulent customers:", len(fraudulent_customers) / len(customer_data) * 100)
