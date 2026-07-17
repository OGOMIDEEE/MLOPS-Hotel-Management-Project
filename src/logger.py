#This file is for a logs sysytem

import logging
import os
from datetime import datetime


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

#setting up the logfile
#How the log file will look like once it is created log_2026-06-12.log
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%M-%d')}.log")

#setup config for the log
#
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',

    #only info and warning messages will be shown
    level=logging.INFO
)

#setup function that is used to initialize our logger
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger