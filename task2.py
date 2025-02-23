# developed by Yehyun Lee
# @ copyright 2025 Yehyun Lee
# built on top of task1.py
# i removed unnecessary comments and added new code to train an autoencoder for customer embeddings
# again, i included some Q&A to explain the code

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.neighbors import NearestNeighbors

# step 1: load customer data
kyc = pd.read_csv('kyc.csv')
kyc_industry_codes = pd.read_csv('kyc_industry_codes.csv')

# ensure industry_code columns are of the same type
kyc['industry_code'] = kyc['industry_code'].astype(str)
kyc_industry_codes['industry_code'] = kyc_industry_codes['industry_code'].astype(str)

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

# step 5: preprocess data for embedding model
customer_data.fillna(0, inplace=True)

categorical_cols = ['country', 'province', 'city', 'industry']
customer_data[categorical_cols] = customer_data[categorical_cols].astype(str)
one_hot_encoder = OneHotEncoder()
encoded_categories = one_hot_encoder.fit_transform(customer_data[categorical_cols]).toarray()

numerical_cols = ['employee_count', 'sales', 'total_amount', 'avg_amount', 'transaction_count', 
                  'unique_transaction_types', 'debit_count', 'credit_count', 'max_transaction', 'min_transaction']
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(customer_data[numerical_cols])

features = np.hstack([scaled_numerical, encoded_categories])

# step 6: train autoencoder for embeddings
input_dim = features.shape[1]
encoding_dim = 16  # embedding size

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(features, features, epochs=50, batch_size=32, shuffle=True)

customer_embeddings = encoder.predict(features)
customer_data['embedding'] = list(customer_embeddings)

# step 7: anomaly detection & bad actor search
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
customer_data['anomaly_score'] = isolation_forest.fit_predict(customer_embeddings)

# nearest neighbor search for similar customers
nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(customer_embeddings)
distances, indices = nbrs.kneighbors(customer_embeddings)
customer_data['nearest_neighbors'] = list(indices)

# save processed data & embeddings
customer_data.to_csv('customer_embeddings.csv', index=False)

# q. what is the purpose of training an autoencoder for customer embeddings?
# a. training an autoencoder allows us to learn low-dimensional representations (embeddings) of customers that capture important patterns in the data.
# these embeddings can be used for anomaly detection and finding similar customers based on their features.

# q. what is autoencoder?
# a. an autoencoder is a type of neural network that learns to
# reconstruct its input data, typically by compressing the input into a
# lower-dimensional representation (encoder) and then reconstructing the original 
# input from the compressed representation (decoder).
# in our case, we use following hyperparameters and settings:
    # input_dim = features.shape[1]
    # encoding_dim = 16  # embedding size
    # input_layer = Input(shape=(input_dim,))
    # encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # decoded = Dense(input_dim, activation='sigmoid')(encoded)
    # autoencoder = keras.Model(input_layer, decoded)
    # encoder = keras.Model(input_layer, encoded)
    # autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.fit(features, features, epochs=50, batch_size=32, shuffle=True)


# what the code does for task2.py:
# 1. loads customer data and transaction data.
# 2. aggregates transaction data by customer.
# 3. merges aggregated transactions with customer data.
# 4. preprocesses data for an embedding model.
# 5. trains an autoencoder to learn embeddings for customers.
# 6. performs anomaly detection using Isolation Forest.
# 7. finds nearest neighbors for each customer based on embeddings.
# 8. saves processed data and embeddings to a CSV file.


# what code did in task1.py:

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


# the main difference between task1.py and task2.py is the approach to customer data analysis.
# task1.py focuses on clustering and anomaly detection using KMeans and Isolation Forest algorithms, respectively.
# task2.py focuses on learning embeddings for customers using an autoencoder and performing anomaly detection with Isolation Forest.
# task2.py also includes a nearest neighbor search to find similar customers based on embeddings.
