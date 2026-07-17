#MODEL TRAINING AND EXPERIMENT TRAINING

import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_function import read_yaml,load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        #Initialize Lightgbm parameters
        self.params_dist = LIGHTGMB_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    #Load and split our data
    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.train_path}")
            test_df = load_data(self.test_path)

            #Ensure the columns are consistent in both of the data
            #split the data
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info(f"Data Split successful for model training")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data", e)
        

    #Train the model
    def train_lgbm(self, X_train,y_train):
        try:
            logger.info("Initializing our model")

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])

            logger.info("Starting our Hyperparamter tuning")

            random_search = RandomizedSearchCV(
                estimator= lgbm_model,
                param_distributions= self.params_dist,
                n_iter= self.random_search_params["n_iter"],
                cv= self.random_search_params["cv"],
                n_jobs= self.random_search_params["n_jobs"],
                verbose= self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring= self.random_search_params["scoring"]
            )

            logger.info("Starting our Hyperparamter tuning")

            random_search.fit(X_train,y_train)

            logger.info("Hyperparamter tuning complted")

            #The best parameters are stored here
            best_params = random_search.best_params_

            #The best model is stored here
            best_lgm_model = random_search.best_estimator_

            logger.info(f"Best Parameters are : {best_params}")

            return best_lgm_model
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to Train Model", e)
        
    
    #Evaluate the accuracy, recall, precision, and f1
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating our model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            #Log all the data results
            logger.info(f"Accuracy score : {accuracy}")
            logger.info(f"Precision score : {precision}")
            logger.info(f"Recall score : {recall}")
            logger.info(f"F1 score : {f1}")

            #return all the metrics
            return {
                "accuracy" : accuracy,
                "precision": precision,
                "recall" : recall,
                "f1" : f1
            }
        
        except Exception as e:
            logger.error(f"Error while evaluate model {e}")
            raise CustomException("Failed to evaluate Model", e)
        
    #Save our model

    def save_model(self, model):
        try:
            #check if the directory first exit for model saving and create it if it doesnt
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("Saving the model")
            #save the model in the output path
            joblib.dump(model, self.model_output_path)

            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save Model", e)
        

    def run(self):
        try:
           with mlflow.start_run():
                logger.info("Starting Our Model Trainig pipeline")

                logger.info("Starting our MLFLOW experimentation")

                #we want to save the dataset that was used to train this particular model

                logger.info("Logging the training and testing dataset to MLFLOw")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train,y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                #log the model output path - basically log the model to MLFLOW
                logger.info("Logging the model to MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and Metrics to MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training successfully completed")

        except Exception as e:
            logger.error(f"Error in model training pipeline {e}")
            raise CustomException("Failed during model training pipeline", e)
        


if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()





