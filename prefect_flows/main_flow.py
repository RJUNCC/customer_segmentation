from prefect import task, Flow, flow
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import BytesIO, StringIO
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, cut_tree
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os

load_dotenv("../.env")

# Initiate connection string, container name, and file (blob) name
connection_string = os.getenv("CSTRING")
container_name = "container1"
blob_name = "raw/data.csv"
output_blob_name = "processed/cleaned_data.parquet"
result_blob_name = "results/final_data.parquet"
model_blob_name = "models/cluster_model.joblib"

@task
def read_csv_data():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    blob_data = blob_client.download_blob().content_as_bytes()
    try:
        df = pd.read_csv(BytesIO(blob_data), encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(blob_data), encoding='ISO-8859-1')
    return df

@task
def clean_data(df):
    from scipy import stats
    import numpy as np

    # Drop NA values
    df = df.dropna()

    # Make date column
    df["Date"] = pd.to_datetime(df["InvoiceDate"])

    # Recency
    df['rank'] = df.sort_values(by=["CustomerID", "Date"]).groupby(by=['CustomerID'])["Date"].rank(method="min").astype(int)
    df_rec = df[df['rank'] == 1]
    df_rec['recency'] = (df_rec['Date'] - pd.to_datetime(min(df_rec['Date']))).dt.days

    # Frequency
    freq = df_rec.groupby("CustomerID")['Date'].count()
    df_freq = pd.DataFrame(freq).reset_index()
    df_freq.columns = ['CustomerID', 'frequency']

    # Merge Frequency with Recency using CustomerID
    rec_freq = df_freq.merge(df_rec, on="CustomerID")
    rec_freq['total'] = rec_freq['Quantity'] * rec_freq['UnitPrice']

    # Monetary Value
    m = rec_freq.groupby('CustomerID')['total'].sum()
    m = pd.DataFrame(m).reset_index()
    m.columns = ['CustomerID', 'monetary_value']

    # Merge Recency and Frequency with Monetary Value using CustomerID
    rfm = m.merge(rec_freq, on='CustomerID')

    # Create new dataframe without CustomerID
    finaldf = rfm[['CustomerID', 'recency', 'frequency', 'monetary_value']]
    new_df = finaldf[['recency', 'frequency', 'monetary_value']]

    # Remove outliers
    z_scores = stats.zscore(new_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = new_df[filtered_entries]
    new_df = new_df.drop_duplicates()
    col_names = ['recency', 'frequency', 'monetary_value']
    features = new_df[col_names]

    # Standardization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.values)
    scaled_features_df = pd.DataFrame(scaled_features, columns=col_names)
    return scaled_features_df

@task
def save_parquet(df):
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    return parquet_buffer.getvalue()

@task
def upload_to_blob(data, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)

@task
def read_parquet_data():
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(output_blob_name)
    blob_data = blob_client.download_blob().readall()
    df = pd.read_parquet(BytesIO(blob_data))
    return df

@task
def segment_data(scaled_features):
    ward_clust = linkage(scaled_features, method="ward", metric="euclidean")
    cluster_labels = cut_tree(ward_clust, n_clusters=2).reshape(-1,)
    scaled_features["Cluster"] = cluster_labels
    return scaled_features, ward_clust

@task
def save_model(model):
    model_buffer = BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    return model_buffer.getvalue()

# @task
# def get_model()

# @task
# def log_model_with_mlflow(model):
#     with mlflow.start_run():
#         mlflow.sklearn.log_model(model, "model")
#         mlflow.log_params({"model_type": "ward_hierarchical_clustering"})
#         mlflow.log_artifact(model_blob_name)

@flow(log_prints=True)
def run_flow():
    df = read_csv_data()
    cleaned_df = clean_data(df)
    cleaned_data_parquet = save_parquet(cleaned_df)
    upload_to_blob(cleaned_data_parquet, output_blob_name)
    
    scaled_features = read_parquet_data()
    segmented_df, model = segment_data(scaled_features)
    final_data_parquet = save_parquet(segmented_df)
    upload_to_blob(final_data_parquet, result_blob_name)
    
    model_data = save_model(model)
    upload_to_blob(model_data, model_blob_name)
    
    # log_model_with_mlflow(model)

if __name__ == "__main__":
    run_flow()
