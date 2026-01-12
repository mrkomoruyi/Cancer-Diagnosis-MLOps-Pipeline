# End-to-end Cancer Diagnosis using Machine Learning and Flask

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A complete MLOps pipeline for cancer diagnosis that demonstrates data ingestion, transformation, model training & evaluation, and a Flask-based web UI for inference.

**Dataset:** This project uses the [Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) from Kaggle.

## Features

- **Reproducible Pipeline:** Modular scripts for ingestion, transformation, and training.
- **Robustness:** Custom exception handling and logging for every step.
- **Web Interface:** Clean Flask UI for real-time predictions.
- **Artifact Management:** Systematically saves preprocessors and models for deployment.
- **Notebook Experiments:** Comprehensive EDA and model experiments.
- **AWS Deployment-ready:** Elastic beanstalk configuration file set up.

## Preview (Flask Web UI)

https://github.com/user-attachments/assets/8d40432d-c4c0-40e3-97ed-a95b05f894ed

## TODO

- Add step-by-step instructions for Elastic beanstalk deployment to README.

## How to run this project

### Prerequisites:

- conda
- Git

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
```

5. Open [http://localhost:5000/predict](http://localhost:5000/predict) in your browser.

## Project structure (high level)

-   **`notebook/`**: contains the raw dataset, exploratory data analysis (EDA), and experimentation notebooks.
-   **`artifacts/`**: stores generated outputs such as raw/processed data files, the trained `model.pkl`, and preprocessor objects.
-   **`src/`**: the core source code for the project:
    -   `components/`: modular scripts for Data Ingestion, Transformation, and Model Training.
    -   `pipeline/`: orchestration scripts for the Training and Prediction pipelines.
    -   `utils.py`, `logger.py`, `exception.py`: common utility functions, custom logging, and exception handling logic.
-   **`templates/`**: HTML files (`index.html`, `home.html`) for the Flask web interface.
-   **`application.py`**: the main entry point for the Flask web application.
-   **`setup.py` & `requirements.txt`**: configuration for project dependencies and package installation.

## Contributing

If you find a bug, please submit an issue using the Issues tab.

If you want to submit a Pull Request, open an issue first and reference the issue in the pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
