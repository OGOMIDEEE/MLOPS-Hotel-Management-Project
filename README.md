# Hotel Reservation Cancellation Prediction

An end-to-end MLOps project that predicts whether a hotel reservation will be cancelled. It covers the full lifecycle: pulling raw data from Google Cloud Storage, preprocessing and feature selection, training a LightGBM model with hyperparameter search (tracked via MLflow), and serving predictions through a Flask web app.

## Features

- **Data ingestion** from a GCP bucket with an automatic train/test split.
- **Data preprocessing** — encoding, skewness handling, and feature selection.
- **Model training** using LightGBM with `RandomizedSearchCV`, with experiments logged to MLflow.
- **Flask web app** with a simple UI ([templates/index.html](templates/index.html)) for entering booking details and getting a live cancellation prediction.


## Project Structure

```
├── application.py           # Flask app entry point
├── config/                  # YAML config, paths, and model hyperparameters
├── src/                     # Core pipeline logic
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── logger.py
│   └── custom_exception.py
├── pipeline/
│   └── training_pipeline.py # Orchestrates ingestion -> processing -> training
├── utils/
│   └── common_function.py
├── notebook/                 # Exploratory analysis
├── static/ & templates/      # Flask frontend assets
├── artifacts/                # Generated data/model artifacts (raw, processed, models)
└── custom_jenkins/
    └── Dockerfile            # Jenkins server image with Docker CLI/engine for CI/CD
```

## Getting Started

### Prerequisites

- Python 3.11
- A GCP service account with access to the Cloud Storage bucket configured in [config/config.yaml](config/config.yaml)

### Installation

```bash
git clone <repo-url>
cd hotel-management-project
python -m venv venv
venv\Scripts\activate        # On Windows
pip install -e .
```

### Configure GCP credentials

```bash
set GOOGLE_APPLICATION_CREDENTIALS=<path-to-your-service-account-key.json>
```

### Run the training pipeline

```bash
python pipeline/training_pipeline.py
```

This downloads the raw data from GCS, splits it into train/test sets, preprocesses it, and trains a LightGBM model, saving the artifact to `artifacts/models/lgbm_model.pkl`.

### Run the web app

```bash
python application.py
```

Then open `http://localhost:5000` in your browser, fill in the reservation details, and get a prediction.

## Configuration

Pipeline behavior (bucket name, train/test ratio, categorical/numerical columns, etc.) is controlled via [config/config.yaml](config/config.yaml). Model hyperparameters and search space live in [config/model_params.py](config/model_params.py).

## CI/CD (Jenkins)

[custom_jenkins/Dockerfile](custom_jenkins/Dockerfile) builds a Jenkins server with the Docker CLI/engine installed, so pipeline jobs can run `docker` commands directly (e.g. build/push the app image).

Build the image:

```bash
cd custom_jenkins
docker build -t jenkins-dind .
```

Run it, mounting the host's Docker socket so Jenkins can talk to your local Docker engine, plus a named volume so Jenkins config/jobs persist:

```bash
docker run -d --name jenkins-dind \
  -p 8080:8080 -p 50000:50000 \
  -v //var/run/docker.sock:/var/run/docker.sock \
  -v jenkins_home:/var/jenkins_home \
  jenkins-dind
```

Get the initial admin password and finish setup at `http://localhost:8080`:

```bash
docker exec -it jenkins-dind cat /var/jenkins_home/secrets/initialAdminPassword
```

## Tech Stack

- **ML**: scikit-learn, LightGBM, imbalanced-learn
- **Experiment tracking**: MLflow
- **Data**: pandas, numpy, Google Cloud Storage
- **Web**: Flask
- **CI/CD**: Jenkins, Docker

