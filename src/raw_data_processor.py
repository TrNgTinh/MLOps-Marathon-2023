import argparse
import logging
import pickle
from pickle import dump
import numpy as np
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from problem_config import ProblemConfig, ProblemConst, get_prob_config
from utils import save_label_encoder, load_label_encoder

from collections import Counter

class LabelEncoderExt(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.most_common_label = None

    def fit(self, data):
        self.classes_ = np.unique(data)
        # Lưu giữ giá trị xuất hiện nhiều nhất trong tập huấn luyện
        label_counts = Counter(self.classes_)
        self.most_common_label = label_counts.most_common(1)[0][0]
        return self

    def transform(self, data):
        # Gán nhãn mới không có trong tập lớp đã khớp thành giá trị mode
        new_data = np.where(np.isin(data, self.classes_), data, self.most_common_label)
        return super().transform(new_data)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
class RawDataProcessor:

    @staticmethod
    def encode_category_features(data, prob_config, categorical_cols=None):
        
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            if df[col].value_counts().ne(1).sum() >= 10:
                name_file_save = "LE_" + str(col) + ".pkl"
                le = load_label_encoder(prob_config.label_encoder / name_file_save)
                df[col] = le.fit_transform(df[col])    
            else:
                name_file_save = "OH_" + str(col) + ".pkl"
                onehot_encoder = load_label_encoder(prob_config.label_encoder / name_file_save)
                encoded_col = onehot_encoder.transform(df[[col]])
                col_names = [col + '_' + str(i) for i in range(encoded_col.shape[1])]
                df_encoded = pd.DataFrame(encoded_col, columns=col_names)
                df = pd.concat([df, df_encoded], axis=1)   
                df.drop(columns=[col], inplace=True)
        return df

    @staticmethod
    def load_models_from_folder(prob_config):
        model_dict = {}
        for filename in os.listdir(prob_config.label_encoder):
            if filename.endswith(".pkl"):
                col_name = filename[:-4]  # Remove the '.pkl' extension
                model_path = os.path.join(prob_config.label_encoder, filename)
                model = load_label_encoder(model_path)
                model_dict[col_name] = model
        return model_dict

    @staticmethod
    def fill_missing_with_mode(df):
        for column in df.columns:
            mode_value = df[column].mode()[0]  # Lấy mode của cột
            df[column].fillna(mode_value, inplace=True)
        return df

    @staticmethod
    def preprocess_categorical_columns(df, model_dict):
        for col, model in model_dict.items():
            col_name = col[3:] 
            if col_name in df.columns:
                if col.startswith("LE_"):
                    df[col_name] = df[col_name].replace('null', model.most_common_label)
                    df[col_name] = model.transform(df[col_name])
                elif col.startswith("OH_"):
                    mode_value = df[col_name].mode().iloc[0]
                    df[col_name] = df[col_name].replace('null', np.nan)
                    df[col_name].fillna(mode_value, inplace=True)
                    if col_name in df.columns:
                        encoded_col = model.transform(df[[col_name]])
                        col_names = [col_name + '_' + str(i) for i in range(encoded_col.shape[1])]
                        df_encoded = pd.DataFrame(encoded_col, columns=col_names)
                        df = pd.concat([df, df_encoded], axis=1)
                        df.drop(columns=[col_name], inplace=True)
        return df

    @staticmethod
    def build_category_features(data, categorical_cols=None):

        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df

    @staticmethod
    def process_raw_data(prob_config: ProblemConfig):
        logging.info("start process_raw_data")
        # read parquet
        training_data = pd.read_parquet(prob_config.raw_data_path)
        #drop duplicates
        training_data = training_data.drop_duplicates()
        #reset index
        training_data = training_data.reset_index(drop=True)
        #encode categorical
        training_data = RawDataProcessor.encode_category_features(
            training_data, prob_config, prob_config.categorical_cols
        )

        selected_features = RawDataProcessor.load_selected_features(prob_config)
        print("selected_features",selected_features )
        if set(training_data['label'].unique()) != {0, 1}:
            # No need to perform label encoding
            label_encoder = LabelEncoder()
            training_data['label'] = label_encoder.fit_transform(training_data['label'])
        
        #split data val, train
        train, dev = train_test_split(
            training_data,
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
        )

        target_col = prob_config.target_col
        train_x = train.drop([target_col], axis=1)
        # Sử dụng phương thức loc để lọc DataFrame
        train_x = train_x.loc[:, selected_features]  
        train_y = train[[target_col]]
        # Sử dụng phương thức loc để lọc DataFrame
        test_x = dev.drop([target_col], axis=1)
        test_x = test_x.loc[:, selected_features]  
        test_y = dev[[target_col]]

        logging.info(f"shape training_data: {train_x.shape}")
        #full data
        full_x = training_data.drop([target_col], axis=1)
        full_y = training_data[[target_col]]
        full_x.to_parquet(prob_config.full_x_path, index=False)
        full_y.to_parquet(prob_config.full_y_path, index=False)

        train_x.to_parquet(prob_config.train_x_path, index=False)
        train_y.to_parquet(prob_config.train_y_path, index=False)
        test_x.to_parquet(prob_config.test_x_path, index=False)
        test_y.to_parquet(prob_config.test_y_path, index=False)

        
        logging.info("finish process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig):
        train_x_path = prob_config.train_x_path
        train_y_path = prob_config.train_y_path
        train_x = pd.read_parquet(train_x_path)
        train_y = pd.read_parquet(train_y_path)
        return train_x, train_y[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig):
        dev_x_path = prob_config.test_x_path
        dev_y_path = prob_config.test_y_path
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_selected_features(prob_config: ProblemConfig):
        with open(prob_config.selected_features, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    RawDataProcessor.process_raw_data(prob_config)
