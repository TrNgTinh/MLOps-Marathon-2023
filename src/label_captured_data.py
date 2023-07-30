import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from raw_data_processor import RawDataProcessor, LabelEncoderExt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(data, prob_config):
    # Separate categorical and numerical columns
    cat_cols = data.select_dtypes(include='object').columns
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Preprocess numerical columns
    num_data = data[num_cols]
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    num_data = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(num_data)), columns=num_data.columns)

    # Preprocess categorical columns
    cat_data = data[cat_cols]
    # Convert categorical columns to strings
    cat_data = cat_data.astype(str)
    encoder = OneHotEncoder()
    cat_data = pd.DataFrame(encoder.fit_transform(cat_data).toarray(), columns=encoder.get_feature_names_out(cat_cols))

    # Concatenate numerical and categorical columns
    processed_data = pd.concat([num_data, cat_data], axis=1)

    return processed_data
def label_captured_data(prob_config: ProblemConfig, threshold):
    train_x = pd.read_parquet(prob_config.full_x_path)
    train_y = pd.read_parquet(prob_config.full_y_path)
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])

    # Preprocess training data
    train_x = preprocess_data(train_x, prob_config)
    np_captured_x = preprocess_data(captured_x, prob_config).to_numpy()

    n_captured = len(np_captured_x)
    n_samples = len(train_x) + n_captured
    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    logging.info("Initialize and fit the clustering model")
    max_clusters = int(n_samples / 10) * len(np.unique(train_y))
    distortions = []
    cluster_range = range(1, max_clusters + 1)
    for n_cluster in cluster_range:
        if n_cluster % 10 == 0:
            print("Current N Cluster:", n_cluster)
        kmeans_model = MiniBatchKMeans(
            n_clusters=n_cluster, random_state=prob_config.random_state, n_init=5
        ).fit(train_x)
        distortions.append(kmeans_model.inertia_)

    logging.info("Find the optimal number of clusters using the Elbow method")
    elbow_index = find_elbow(distortions)
    n_cluster_optimal = cluster_range[elbow_index]
    logging.info(f"Optimal number of clusters: {n_cluster_optimal}")

    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster_optimal, random_state=prob_config.random_state, n_init=5
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info("Assign new labels to the new data based on the labels of the original data in each cluster")
    new_labels = []
    for i in range(n_cluster_optimal):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # Calculate the confidence score for the cluster
            cluster_confidence = np.max(kmeans_model.transform(np_captured_x)[:, i])

            if cluster_confidence > threshold:
                # For a linear regression problem, use the mean of the labels as the new label
                # For a logistic regression problem, use the mode of the labels as the new label
                if ml_type == "regression":
                    new_labels.append(np.mean(cluster_labels.flatten()))
                else:
                    new_labels.append(
                        np.bincount(cluster_labels.flatten().astype(int)).argmax()
                    )

    approx_label = [new_labels[c] if cluster_confidence > threshold else None for c, cluster_confidence in zip(kmeans_clusters, np.max(kmeans_model.transform(np_captured_x), axis=1))]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)

def find_elbow(distortions):
    # Calculate the change in distortions between consecutive clusters
    distortions_diff = np.diff(distortions)

    # Calculate the second derivative of distortions
    distortions_diff2 = np.diff(distortions_diff)

    # Find the index of the elbow point
    elbow_index = np.argmax(distortions_diff2) + 2  # Add 2 to account for the two differences taken

    return elbow_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--threshold", type=float, default=0.5)  # Add a threshold argument
    args = parser.parse_args()

    # Disable the FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    prob_config = get_prob_config(args.phase_id, args.prob_id)

    model_category = RawDataProcessor.load_models_from_folder(prob_config)
    selected_features = RawDataProcessor.load_selected_features(prob_config)
    
    label_captured_data(prob_config, args.threshold)  # Pass the threshold argument
