import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_function import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

#
logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        

        #read the file that is passed
        self.config = read_yaml(config_path)

        #check and see if the preprocessed direcory is created
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    #Process the data
    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data processng")

            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]


            label_encoder = LabelEncoder()
            mappings={}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])

                mappings[col] = {label:code for label, code in zip(label_encoder.classes_ , label_encoder.transform(label_encoder.classes_))}

            logger.info("Lable Mappings are : ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")


            #Skewness Handling
            logger.info("Doing Skewness Handling")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())

            #apply log transformation to fix the skewness[basically checking where there is askewness greater than 5]
            for column in skewness[skewness>skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocessing data", e)
        
    
    #imbalanced data
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced data")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_resampled , y_resampled = smote.fit_resample(X,y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info(f"Data balance successful")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data data", e)
        

    #Feature Selection
    def select_features(self, df):
        try:
            logger.info("Starting Feature selection step")

            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance':feature_importance
            })

            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]

            #Instead of writing 10 like we have in the notebook, we use the value we passed in config.yaml file

            top_10_features = top_features_importance_df["feature"].head(num_features_to_select).values

            logger.info(f"Features selected: {top_10_features}")

            top_10_df = df[top_10_features.tolist() + ["booking_status"]]

            logger.info("Feature selection completed successfully")

            return top_10_df

        except Exception as e:
            logger.error(f"Error during Feature selection step {e}")
            raise CustomException("Error while Feature selection ", e)
        

    #Save data method
    def save_data(self,df, file_path):
        try:
            logger.info("Saving our data in processed folder")

            df.to_csv(file_path, index=False)

            logger.info(f"Data saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during Saving data step {e}")
            raise CustomException("Error while Saving data ", e)
        
    #combining all our steps to run it
    def process(self):
        try:
            logger.info("Loading data from Raw Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            #preprocessing step

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            #balancing dataset
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            #select top feature[the features selected for train will be selected for test also]
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            #save data
            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)

            logger.info(f"Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error during Preprocessing Pipeline {e}")
            raise CustomException("Error while data Preprocessing Pipeline ", e)


if __name__=="__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
    