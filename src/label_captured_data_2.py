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
from joblib import load

# Function to load the pre-trained model
def load_pretrained_model(model_path):
    return load(model_path)

def label_captured_data(prob_config: ProblemConfig, threshold):
    train_x = pd.read_parquet(prob_config.full_x_path)
    train_y = pd.read_parquet(prob_config.full_y_path)
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_x = pd.concat([captured_x, captured_data])
    print("captured_x.shape", captured_x.shape)
    # Drop duplicates
    captured_x = captured_x.drop_duplicates()
    captured_x = captured_x.reset_index(drop=True)

    model_category = RawDataProcessor.load_models_from_folder(prob_config)

    training_data = RawDataProcessor.preprocess_categorical_columns(
            captured_x, model_category
        )
    selected_features = RawDataProcessor.load_selected_features(prob_config)

    training_data = training_data.loc[:, selected_features]  

    # Load the pre-trained model
    model = load_pretrained_model('/home/ubuntu/MLOps-Marathon-2023/model_prob_2.pkl')

    logging.info("Predict the probabilities for the new data using the pre-trained model")

    predicted_probabilities = model.predict_proba(training_data)
    print("predicted_probabilities shape", predicted_probabilities.shape)
    logging.info("Assign new labels to the new data based on the predicted probabilities")
    new_labels = []
    for i, probabilities in enumerate(predicted_probabilities):
        max_probability = np.max(probabilities)
        if max_probability > threshold:
            # For a linear regression problem, use the predicted value as the new label
            # For a classification problem, use the class with the highest probability as the new label
            if prob_config.ml_type == "regression":
                new_labels.append(probabilities[0])  # Assuming your regression model returns a single value
            else:
                new_labels.append(np.argmax(probabilities))
        else:
            new_labels.append(None)
    
    # Lấy chỉ mục của tất cả các giá trị trong new_labels khác None
    non_none_indices = [i for i, label in enumerate(new_labels) if label is not None]

    # Lọc lại dữ liệu huấn luyện (training data) theo các chỉ mục không None
    filtered_training_data = training_data.iloc[non_none_indices]
    filtered_labels = [new_labels[i] for i in non_none_indices]

    approx_label_df = pd.DataFrame(filtered_labels, columns=[prob_config.target_col])
    filtered_training_data.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--threshold", type=float, default=0.97)  # Add a threshold argument
    args = parser.parse_args()

    # Disable the FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    prob_config = get_prob_config(args.phase_id, args.prob_id)

    model_category = RawDataProcessor.load_models_from_folder(prob_config)
    selected_features = RawDataProcessor.load_selected_features(prob_config)
    
    label_captured_data(prob_config, args.threshold)  # Pass the threshold argument
