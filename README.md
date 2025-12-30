# End-to-end cancer diagnosis using Machine learning and Flask

## Overview

A complete MLOps pipeline for cancer diagnosis that demonstrates data ingestion, transformation, model training & evaluation, and a Flask-based web UI for inference.

## Features

-   Exploratory data analysis.
-   Data ingestion and preprocessing.
-   Model training and evaluation.
-   Simple Flask web UI for inference.
-   Scripts for reproducible experiments and MLOps steps.

## Preview (Flask Web UI)

https://github.com/user-attachments/assets/8d40432d-c4c0-40e3-97ed-a95b05f894ed

## How to run this project

### Prerequisites:

-   conda
-   Git (recommended)

### Run:

```bash
# 1. Clone repository
git clone https://github.com/mrkomoruyi/Cancer-Diagnosis-MLOps-Pipeline.git
cd Cancer-Diagnosis-MLOps-Pipeline

# 2. Create env and install
conda create -p venv python==3.12 -y
conda activate venv/
pip install -r requirements.txt

# 3. Run the training script
python src/pipeline/train_pipeline.py

# 4. Run the web app
python application.py

# 5. Open http://localhost:5000/predict in your browser.
```

## Project structure (high level)

-   notebook/ - raw dataset, EDA and experiments
-   artifacts/ - raw and processed datasets, saved model and preprocessor artifacts
-   src/components - training, evaluation, preprocessing scripts
-   src/pipeline - prediction and training pipeline scripts
-   application.py - Flask web application
-   requirements.txt

## Contributing

If you find a bug, please submit an issue using the Issues tab.

If you want to submit a Pull Request, open an issue first and reference the issue in the pull request.
