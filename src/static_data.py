import argparse
import logging
import pickle
import os
import time
from collections import Counter
from utils import save_label_encoder, load_label_encoder

from problem_config import ProblemConfig, ProblemConst, get_prob_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from scipy.stats import randint as sp_randint
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import OrdinalEncoder
import warnings
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

SEED = 42

np.random.seed(SEED)  # Set the seed for numpy

def train_and_evaluate_model(X_train, y_train):
    # Split the data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

    # Create your model (use LightGBM or any other classifier you want)
    model = LGBMClassifier(random_state=SEED)  # Replace LGBMClassifier with your chosen classifier

    # Train the model
    model.fit(X_train_split, y_train_split)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the model using the desired evaluation metric (e.g., accuracy, precision, recall, etc.)
    # Use the appropriate metric for your problem
    metric = accuracy_score(y_val, y_pred)  # Replace accuracy_score with your chosen metric

    return metric

class LabelEncoderExt(LabelEncoder):
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


class OneHotEncoderClass:
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

    def fit_transform(self, df):
        for column in self.columns:
            df[column] = df[column].astype('category')
            encoded_column = self.encoder.fit_transform(df[column].values.reshape(-1, 1))
            df = pd.concat([df, pd.DataFrame(encoded_column.toarray(), columns=[column + '_' + str(i) for i in range(encoded_column.shape[1])])], axis=1)
        return df

class ModelEvaluator:
    def __init__(self, models, X_train_scaled, y_train, output_file):
        self.models = models
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.metrics = ['precision']
        self.scores = self.evaluate_models()
        self.output_file = output_file


    def evaluate_models(self):
        scores = {}
        for name, model in self.models.items():
            scores[name] = {}
            for metric in self.metrics:
                if metric == 'precision':
                    scorer_func = make_scorer(precision_score, average='weighted')
                elif metric == 'recall':
                    scorer_func = make_scorer(recall_score, average='weighted')
                scores[name][metric] = cross_val_score(model, self.X_train_scaled, self.y_train, cv=10, scoring=scorer_func, error_score="raise")
        return scores

    def print_scores(self, model_name, best_params, best_score, output_file):
        with open(output_file, 'a') as f:
            f.write(f'{model_name} Model Validation\n')
            f.write(f'Best Parameters: {best_params}\n')
            f.write(f'Best Score: {round(best_score * 100, 2)}%\n\n')

    def get_scores_dataframe(self):
        scores_df = pd.DataFrame(self.scores).swapaxes("index", "columns") * 100
        return scores_df

    def optimize_models(self):
        # Định nghĩa các giá trị siêu tham số để tối ưu
        param_dist = {
            'n_estimators': np.arange(100, 1300, 50),  # Số cây trong mô hình
            'learning_rate': [0.01, 0.1],  # Tỷ lệ học
            'num_leaves': sp_randint(5, 50),  # Số lượng lá cây
            'max_depth': sp_randint(3, 5),  # Độ sâu tối đa của cây
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Tỷ lệ mẫu con
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Tỷ lệ tính năng trong cây
            'reg_alpha': [0.2, 0.3, 0.4],  # L1 regularization
            'reg_lambda': [ 0.2, 0.3, 0.4],  # L2 regularization
        }

        for name, model in self.models.items():
            # Khởi tạo RandomizedSearchCV với mô hình và các tham số đã định nghĩa
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter= 50,  # Số lượng thử nghiệm
                cv=5,  # Số lượng fold cross-validation
                scoring='accuracy',  # Metric để đánh giá mô hình
                random_state=SEED,
                verbose=1,
                n_jobs=-1  # Sử dụng tất cả các core của máy tính để chạy song song các thử nghiệm
            )

            # Thực hiện tìm kiếm tối ưu hóa ngẫu nhiên
            random_search.fit(self.X_train_scaled, self.y_train)

            # Cập nhật mô hình đã tối ưu
            self.models[name] = random_search.best_estimator_

            # Gọi hàm để in thông tin tối ưu hóa và kết quả
            self.print_scores(name, random_search.best_params_, random_search.best_score_, self.output_file)


class StaticRawData:
    @staticmethod
    def static_raw_data(prob_config: ProblemConfig):
        logging.info("start static_data")

        # Read parquet
        #training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data = pd.read_parquet('/mnt/d/MLops/Competition/mlops-mara-sample-public/data/raw_data/phase-3/prob-1/raw_train_combine.parquet')
        training_data = training_data.drop_duplicates()
        training_data = training_data.reset_index(drop=True)
        df_x = training_data.drop("label", axis=1)
        df_y = training_data['label']

        if set(pd.Series(df_y).unique()) != {0, 1}:
            # No need to perform label encoding
            label_encoder = LabelEncoder()
            df_y = label_encoder.fit_transform(df_y)
            joblib.dump(label_encoder, prob_config.label_encoder / 'multi_class.pkl')

        print(df_x.shape)
        # Encode the selected columns
        for col in df_x.select_dtypes("object"):
            # Encoder For Cat Data
            if df_x[col].value_counts().ne(1).sum() >= 2:
                le = LabelEncoderExt()
                le.fit(df_x[col])
                df_x[col] = le.fit_transform(df_x[col])
                #save model
                name_file_save = "LE_" + str(col) + ".pkl"
                save_label_encoder(le, prob_config.label_encoder / name_file_save )

            #elif df_x[col].value_counts().ne(1).sum() >= 2:  # Use OrdinalEncoder for columns with more than 2 unique categories
            #    oe = OrdinalEncoder()
            #    df_x[col] = oe.fit_transform(df_x[[col]])
            #    # save model
            #    name_file_save = "OE_" + str(col) + ".pkl"
            #    save_label_encoder(oe, prob_config.label_encoder / name_file_save)

            else:
                onehot_encoder = OneHotEncoder(drop='first', sparse=False)
                encoded_col = onehot_encoder.fit_transform(df_x[[col]])
                col_names = [col + '_' + str(i) for i in range(encoded_col.shape[1])]
                df_encoded = pd.DataFrame(encoded_col, columns=col_names)
                df_x = pd.concat([df_x, df_encoded], axis=1)
                
                df_x.drop(columns=[col], inplace=True)
                #save model
                name_file_save = "OH_" + str(col) + ".pkl"
                save_label_encoder(onehot_encoder, prob_config.label_encoder / name_file_save )
                
        #print("df_x.isna().sum().sum()", df_x.isna().sum())
        print(df_x.shape)
        scaler = StandardScaler()
        cols = df_x.columns

        ## Transform the data
        df_x_scaled = pd.DataFrame(scaler.fit_transform(df_x), columns=cols)
        estimator = LGBMClassifier(random_state=SEED)  # Set the random state for RandomForestClassifier
        selector = RFE(estimator, n_features_to_select = 28)
        selector.fit(df_x_scaled, df_y)

        df_x_scaled = selector.transform(df_x_scaled)

        feature_map = zip(selector.get_support(), df_x.columns)
        selected_features = [v for i, v in feature_map if i]
        
        logging.info(f"selected_features: {selected_features}")

        # Perform feature selection loop to find the optimal number of features
        #max_num_features = df_x_scaled.shape[1]  # Maximum number of features to try
        #best_metric = 0  # Initialize the best evaluation metric
        #best_num_features = 0  # Initialize the number of features for the best metric
#
        #for num_features in range(15, max_num_features + 1):
        #    logging.info(f"Trying {num_features} features...")
        #    
        #    # Use RFE to select the top num_features features
        #    estimator = LGBMClassifier(random_state=SEED)  # Set the random state for LightGBM Classifier
        #    selector = RFE(estimator, n_features_to_select=num_features)
        #    selector.fit(df_x_scaled, df_y)
#
        #    # Get the selected features
        #    selected_features = df_x_scaled.columns[selector.support_].tolist()
#
        #    # Train and evaluate your model using the selected features, and measure the time
        #    start_time = time.time()
        #    metric = train_and_evaluate_model(df_x_scaled[selected_features], df_y)
        #    end_time = time.time()
        #    elapsed_time = end_time - start_time
        #    logging.info(f"Metric for {num_features} features: {metric}, Training Time: {elapsed_time} seconds")
#
        #    if metric > best_metric:
        #        best_metric = metric
        #        best_num_features = num_features
#
        #logging.info(f"Best number of features: {best_num_features}")
        ## Now you have the best number of features for your model
        ## You can use this number to train your final model with the selected features
        #selected_features = df_x_scaled.columns[:best_num_features].tolist()

        #logging.info(f"selected_features: {selected_features}")

        # Kiểm tra và xoá feature3 khỏi danh sách selected_features (nếu nó có trong đó)
        #if 'feature3' in selected_features:
        #    selected_features.remove('feature3')

        # Save selected features to a pickle file
        selected_features_path = prob_config.selected_features
        with open(selected_features_path, 'wb') as f:
            pickle.dump(selected_features, f)
        # Train model
        # Define models
        models = {
            'LightGBM Classifier': LGBMClassifier(random_state=SEED)
        }
        output_file = prob_config.score_model
        evaluator = ModelEvaluator(models, df_x_scaled, df_y, output_file)
        # Optimize models
        evaluator.optimize_models()
        return selected_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE3)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    StaticRawData.static_raw_data(prob_config)
