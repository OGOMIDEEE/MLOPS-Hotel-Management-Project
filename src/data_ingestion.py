#Data Ingestion process

#to link your credentials = 
# set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\mide\Downloads\mlops-project-keys\project-8e173095-fc94-4412-86e-079354100406.json
# Bascially we have given access of the storage an dbucket so we can extarct the data from the bucket and use it here

import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_function import  read_yaml


logger =  get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        #setting up the raw folder- to store the end process
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and file is {self.file_name}")

    #download csv from gcp bucket
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            #downlaod the file to the raw file path
            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file is successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error("Error while downloading the csv file")
            raise CustomException("Failed to download CSV file ", sys)
        
    
        
    #Split data
    def split_data(self):
        try:
            logger.info("Starting the splitting process")

            #Read the data from raw file path
            data = pd.read_csv(RAW_FILE_PATH)

            #split the data into train and test data - 
            train_data, test_data = train_test_split(data , test_size=1 - self.train_test_ratio, random_state=42)

            #convert the data from dataFrame to CSV and store it in the file path
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while splitting the data")
            raise CustomException(f"Failed to split data into training and test sets ", sys)
        
    ##Run method to help us run the whole process
    def run(self):
        try:
            logger.info(f"Starting data Ingestio  Process")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data Ingestion completed successfully")

        except CustomException as ce:
            #str(ce) - text representation of the error so it can be stored easily on the logger
            logger.error(f"CustomException : {str(ce)}")

        finally:
            logger.info("Data Ingestion Completed")


##To run the file - python src/data_ingestion.py
if __name__ == "__main__":

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
