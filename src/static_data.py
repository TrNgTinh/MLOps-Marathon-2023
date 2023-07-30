import argparse
import logging
import pickle
import os
from collections import Counter
from utils import save_label_encoder, load_label_encoder

from problem_config import ProblemConfig, ProblemConst, get_prob_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

SEED = 42

np.random.seed(SEED)  # Set the seed for numpy

class LabelEncoderExt(LabelEncoder):
    def fit(self, data):
        self.classes_ = np.unique(data)
        # Lưu giữ giá trị xuất hiện nhiều nhất trong tập huấn luyện
        label_counts = Counter(self.classes_)
        self.most_common_label = label_counts.most_common(1)[0][0]
        print("self.most_common_label", self.most_common_label)
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
    def __init__(self, models, X_train_scaled, y_train):
        self.models = models
        self.X_train_scaled = X_train_scaled
        self.y_train = y_train
        self.metrics = ['precision']
        self.scores = self.evaluate_models()

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

    def print_scores(self, output_file):
        with open(output_file, 'w') as f:
            for name in self.models:
                f.write(f'{name} Model Validation\n')
                for scorer in self.metrics:
                    mean = round(np.mean(self.scores[name][scorer]) * 100, 2)
                    stdev = round(np.std(self.scores[name][scorer]) * 100, 2)
                    f.write(f"Mean {scorer}:\n")
                    f.write(f"{mean}% +- {stdev}\n")
                f.write('\n')

    def get_scores_dataframe(self):
        scores_df = pd.DataFrame(self.scores).swapaxes("index", "columns") * 100
        return scores_df

class StaticRawData:
    @staticmethod
    def static_raw_data(prob_config: ProblemConfig):
        logging.info("start static_data")

        # Read parquet
        training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data = training_data.drop_duplicates()
        training_data = training_data.reset_index(drop=True)
        df_x = training_data.drop("label", axis=1)
        df_y = training_data['label']

        if set(df_y.unique()) != {0, 1}:
            # No need to perform label encoding
            label_encoder = LabelEncoder()
            df_y = label_encoder.fit_transform(df_y)

        print(df_x.shape)
        # Encode the selected columns
        for col in df_x.select_dtypes("object"):
            # Encoder For Cat Data
            if df_x[col].value_counts().ne(1).sum() >= 10:
                le = LabelEncoderExt()
                le.fit(df_x[col])
                df_x[col] = le.fit_transform(df_x[col])
                #save model
                name_file_save = "LE_" + str(col) + ".pkl"
                save_label_encoder(le, prob_config.label_encoder / name_file_save )
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
                
        ##print("df_x.isna().sum().sum()", df_x.isna().sum())
        #print(df_x.shape)
        #scaler = StandardScaler()
        #cols = df_x.columns
#
        ### Transform the data
        #df_x_scaled = pd.DataFrame(scaler.fit_transform(df_x), columns=cols)
        #estimator = RandomForestClassifier(random_state=SEED)  # Set the random state for RandomForestClassifier
        #selector = RFE(estimator)
        #selector.fit(df_x_scaled, df_y)
#
        #df_x_scaled = selector.transform(df_x_scaled)
#
        #feature_map = zip(selector.get_support(), df_x.columns)
        #selected_features = [v for i, v in feature_map if i]
#
        #logging.info(f"selected_features: {selected_features}")
#
        ## Save selected features to a pickle file
        #selected_features_path = prob_config.selected_features
        #with open(selected_features_path, 'wb') as f:
        #    pickle.dump(selected_features, f)
        ## Train model
        ## Define models
        #models = {
        #    'XGBoost Classifier': XGBClassifier(eval_metric="logloss", random_state=SEED),
        #    'LightGBM Classifier': LGBMClassifier(random_state=SEED)
        #}
        #evaluator = ModelEvaluator(models, df_x_scaled, df_y)
        #output_file = prob_config.score_model
        ## Write the scores to the output file
        #evaluator.print_scores(output_file)
        #return selected_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    StaticRawData.static_raw_data(prob_config)
