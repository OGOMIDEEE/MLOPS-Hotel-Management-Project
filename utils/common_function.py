#creating a function for reading YAML files because we will read the yaml files at various steps such as data processing, data ingestion


import os
import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml


logger = get_logger(__name__)

#use this function to read the yaml files
def read_yaml(file_path):
    try:
        #checking if the config.yaml file isnt there
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given location")
        
        #If file is in the path - then open it
        with open(file_path, "r") as yaml_file:
            #all the content of the file will be stored in the variable config
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the yaml file")
            return config
        

    except Exception as e:
        logger.error("Error while reading YAML file ")
        raise CustomException("Failed to read YAML file", sys)
    

##Once the ML has been completed in the notebook
#function to load

def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        raise  CustomException("Failed to load data", e)