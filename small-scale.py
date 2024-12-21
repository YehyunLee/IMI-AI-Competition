import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# Step 1: Load Customer Data
kyc = pd.read_csv('kyc.csv')
kyc_industry_codes = pd.read_csv('kyc_industry_codes.csv')

# Ensure industry_code columns are of the same type
kyc['industry_code'] = kyc['industry_code'].astype(str)
kyc_industry_codes['industry_code'] = kyc_industry_codes['industry_code'].astype(str)

# Merge KYC data with industry codes
kyc = kyc.merge(kyc_industry_codes, on='industry_code', how='left')

# Load Transaction Data
cheque = pd.read_csv('cheque.csv')
wire = pd.read_csv('wire.csv')
emt = pd.read_csv('emt.csv')
eft = pd.read_csv('eft.csv')
card = pd.read_csv('card.csv')
abm = pd.read_csv('abm.csv')

# Step 2: Combine Transaction Data with a Transaction Type Column
cheque['transaction_type'] = 'cheque'
wire['transaction_type'] = 'wire'
emt['transaction_type'] = 'emt'
eft['transaction_type'] = 'eft'
card['transaction_type'] = 'card'
abm['transaction_type'] = 'abm'

transactions = pd.concat([cheque, wire, emt, eft, card, abm], ignore_index=True)

# Step 3: Aggregate Transaction Data by Customer
agg_transactions = transactions.groupby('customer_id').agg(
    total_amount=('amount_cad', 'sum'),
    avg_amount=('amount_cad', 'mean'),
    transaction_count=('transaction_type', 'count'),
    unique_transaction_types=('transaction_type', 'nunique'),
    debit_count=('debit_credit', lambda x: (x == 'debit').sum()),
    credit_count=('debit_credit', lambda x: (x == 'credit').sum()),
    max_transaction=('amount_cad', 'max'),
    min_transaction=('amount_cad', 'min')
).reset_index()

# Step 4: Merge Aggregated Transactions with Customer Data
customer_data = kyc.merge(agg_transactions, on='customer_id', how='left')

# Step 5: Preprocess Data for Clustering
# Handle missing values
customer_data.fillna(0, inplace=True)

# Encode categorical variables
categorical_cols = ['country', 'province', 'city', 'industry']
# Q. Why specifically these columns?
# A. These columns are categorical variables that need to be one-hot encoded for clustering.
customer_data[categorical_cols] = customer_data[categorical_cols].astype(str)
one_hot_encoder = OneHotEncoder()
encoded_categories = one_hot_encoder.fit_transform(customer_data[categorical_cols]).toarray()
# Q. What is the purpose of one-hot encoding?
# A. One-hot encoding is used to convert categorical variables into numerical format for machine learning models.

# Standardize numerical variables
numerical_cols = ['employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 
                  'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction']
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(customer_data[numerical_cols])
# Q. Why do we need to standardize numerical variables?
# A. Standardizing numerical variables ensures that all features have the same scale, which is important for clustering algorithms.
# For example, if one feature has a range of 0-1 and another feature has a range of 0-1000, the clustering algorithm may give more weight to the feature with a larger range.

# Combine all features
features = np.hstack([scaled_numerical, encoded_categories])
# Q. Why do we need to combine all features?
# A. Combining all features into a single array allows us to use them as input for clustering algorithms.

# Step 6: Dimensionality Reduction (Optional)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
# Q. Why do we need to reduce the dimensionality of the data?
# A. Reducing the dimensionality of the data can help visualize the data in a lower-dimensional space and potentially improve the performance of clustering algorithms.

# Step 7: Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(features)
# Q. Why do we need to use KMeans clustering?
# A. KMeans clustering is a popular clustering algorithm that can group similar data points together based on their features.

# Step 8: Anomaly Detection
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
customer_data['anomaly_score'] = isolation_forest.fit_predict(features)
# Q. Why do we need to use Isolation Forest for anomaly detection?
# A. Isolation Forest is an effective algorithm for detecting anomalies in data by isolating them in the feature space.

# Visualization of Clusters and Anomalies
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=customer_data['cluster'], cmap='viridis', label='Cluster')
plt.title('Customer Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')
plt.show()
# Q. What is Feature 1 here?
# A. Feature 1 is the first principal component obtained from PCA, which is a linear combination of the original features that captures the most variance in the data.
# More specifically, it is the first column of the reduced_features array, containing the transformed data in a lower-dimensional space.
# Using columns like 'employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# Q2. What is Feature 2 here?
# A2. Feature 2 is the second principal component obtained from PCA, which is another linear combination of the original features that captures the second most variance in the data.
# More specifically, it is the second column of the reduced_features array, containing the transformed data in a lower-dimensional space.
# Using columns like 'employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# Visualization of Anomalies
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=customer_data['anomaly_score'], cmap='coolwarm', label='Anomaly')
plt.title('Anomaly Detection')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Anomaly Score')
plt.show()
# Q. What is the purpose of the anomaly score?
# A. The anomaly score indicates the degree of anomaly for each data point, with lower scores indicating more normal data points and higher scores indicating potential anomalies.

# Q. What is PCA Feature 1 and PCA Feature 2?
# A. PCA Feature 1 and PCA Feature 2 are the two principal components obtained from PCA, which are linear combinations of the original features that capture the most variance in the data.
# In this case, they represent the transformed data in a lower-dimensional space that can be visualized for clustering and anomaly detection.
# Using columns like 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# Save Processed Data
customer_data.to_csv('processed_customer_data.csv', index=False)