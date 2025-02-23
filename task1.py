# developed by Yehyun Lee
# @ copyright 2025 Yehyun Lee
# i included some Q&A to explain the code

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# step 1: load customer data
kyc = pd.read_csv('kyc.csv')
kyc_industry_codes = pd.read_csv('kyc_industry_codes.csv')

# ensure industry_code columns are of the same type
kyc['industry_code'] = kyc['industry_code'].astype(str)
kyc_industry_codes['industry_code'] = kyc['industry_code'].astype(str)

# merge KYC data with industry codes
kyc = kyc.merge(kyc_industry_codes, on='industry_code', how='left')

# load transaction data
cheque = pd.read_csv('cheque.csv')
wire = pd.read_csv('wire.csv')
emt = pd.read_csv('emt.csv')
eft = pd.read_csv('eft.csv')
card = pd.read_csv('card.csv')
abm = pd.read_csv('abm.csv')

# step 2: combine transaction data with a transaction type column
cheque['transaction_type'] = 'cheque'
wire['transaction_type'] = 'wire'
emt['transaction_type'] = 'emt'
eft['transaction_type'] = 'eft'
card['transaction_type'] = 'card'
abm['transaction_type'] = 'abm'

transactions = pd.concat([cheque, wire, emt, eft, card, abm], ignore_index=True)

# step 3: aggregate transaction data by customer
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

# step 4: merge aggregated transactions with customer data
customer_data = kyc.merge(agg_transactions, on='customer_id', how='left')

# step 5: preprocess data for clustering
# handle missing values
customer_data.fillna(0, inplace=True)

# encode categorical variables
categorical_cols = ['country', 'province', 'city', 'industry']
# q. why specifically these columns?
# a. these columns are categorical variables that need to be one-hot encoded for clustering.
customer_data[categorical_cols] = customer_data[categorical_cols].astype(str)
one_hot_encoder = OneHotEncoder()
encoded_categories = one_hot_encoder.fit_transform(customer_data[categorical_cols]).toarray()
# q. what is the purpose of one-hot encoding?
# a. one-hot encoding is used to convert categorical variables into numerical format for machine learning models.

# standardize numerical variables
numerical_cols = ['employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 
                  'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction']
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(customer_data[numerical_cols])
# q. why do we need to standardize numerical variables?
# a. standardizing numerical variables ensures that all features have the same scale, which is important for clustering algorithms.
# for example, if one feature has a range of 0-1 and another feature has a range of 0-1000, the clustering algorithm may give more weight to the feature with a larger range.

# combine all features
features = np.hstack([scaled_numerical, encoded_categories])
# q. why do we need to combine all features?
# a. combining all features into a single array allows us to use them as input for clustering algorithms.

# step 6: dimensionality reduction (optional)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)
# q. why do we need to reduce the dimensionality of the data?
# a. reducing the dimensionality of the data can help visualize the data in a lower-dimensional space and potentially improve the performance of clustering algorithms.

# step 7: clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(features)
# q. why do we need to use KMeans clustering?
# a. KMeans clustering is a popular clustering algorithm that can group similar data points together based on their features.

# step 8: anomaly detection
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
customer_data['anomaly_score'] = isolation_forest.fit_predict(features)
# q. why do we need to use Isolation Forest for anomaly detection?
# a. Isolation Forest is an effective algorithm for detecting anomalies in data by isolating them in the feature space.

# visualization of clusters and anomalies
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=customer_data['cluster'], cmap='viridis', label='Cluster')
plt.title('Customer Clusters')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')
plt.show()
# q. what is Feature 1 here?
# a. Feature 1 is the first principal component obtained from PCA, which is a linear combination of the original features that captures the most variance in the data.
# more specifically, it is the first column of the reduced_features array, containing the transformed data in a lower-dimensional space.
# using columns like 'employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# q2. what is Feature 2 here?
# a2. Feature 2 is the second principal component obtained from PCA, which is another linear combination of the original features that captures the second most variance in the data.
# more specifically, it is the second column of the reduced_features array, containing the transformed data in a lower-dimensional space.
# using columns like 'employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# visualization of anomalies
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=customer_data['anomaly_score'], cmap='coolwarm', label='Anomaly')
plt.title('Anomaly Detection')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Anomaly Score')
plt.show()
# q. what is the purpose of the anomaly score?
# a. the anomaly score indicates the degree of anomaly for each data point, with lower scores indicating more normal data points and higher scores indicating potential anomalies.

# q. what is PCA Feature 1 and PCA Feature 2?
# a. PCA Feature 1 and PCA Feature 2 are the two principal components obtained from PCA, which are linear combinations of the original features that capture the most variance in the data.
# in this case, they represent the transformed data in a lower-dimensional space that can be visualized for clustering and anomaly detection.
# using columns like 'total_amount', 'avg_amount', 'transaction_count', 'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction' as features for clustering and anomaly detection.

# save processed data
customer_data.to_csv('processed_customer_data.csv', index=False)

# explanation processed_customer_data.csv
# the processed_customer_data.csv file contains the customer data with additional columns for clustering and anomaly detection.
# the 'cluster' column indicates the cluster assignment for each customer based on the KMeans clustering algorithm.
# the 'anomaly_score' column indicates the anomaly score for each customer based on the Isolation Forest algorithm.
# anomaly_score values close to -1 indicate potential anomalies, while values close to 1 indicate normal data points.


# what does the code does:
# 1. load customer data and transaction data.
# 2. merge transaction data with customer data.
# 3. aggregate transaction data by customer.
# 4. preprocess the data for clustering.
# 5. combine all features for clustering.
# 6. perform dimensionality reduction using PCA.
# 7. cluster customers using KMeans clustering.
# 8. detect anomalies using Isolation Forest.
# 9. visualize clusters and anomalies using PCA features.
# 10. save the processed customer data to a CSV file.
