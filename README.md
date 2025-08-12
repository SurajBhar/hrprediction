# Hotel Reservation Cancellation Prediction — From Notebook to Production on GCP

Cancellations hurt occupancy forecasts and revenue. If we can predict, at booking time, whether a reservation is likely to be canceled, the hotel can adjust inventory, pricing, and outreach more intelligently. This project takes a real-world dataset of hotel reservations and turns it into a production web app that scores bookings in real time.

![Deployment Flow Diagram/ CI/CD Pipeline](resources/project_flow_2.png)

**Goal of this project:** Predict whether a hotel reservation will be canceled at booking time so the business can manage overbooking and revenue risk.

![Hotel Reservation Cancellation Prediction](resources/crun_negative.png)

**High-level outcome:** In offline experiments, a Random Forest delivered the best accuracy but produced a **~168 MB** artifact—too heavy for fast, low-cost serving. I deployed **LightGBM** instead: nearly identical accuracy with a much smaller model footprint, which lowers container size, startup latency, and Cloud Run costs.

---

## Table of Contents

- [Hotel Reservation Cancellation Prediction — From Notebook to Production on GCP](#hotel-reservation-cancellation-prediction--from-notebook-to-production-on-gcp)
  - [The experimentation notebook: from EDA to model choice](#the-experimentation-notebook-from-eda-to-model-choice)
    - [EDA highlights](#eda-highlights)
    - [Feature engineering & preprocessing (prototyped)](#feature-engineering--preprocessing-prototyped)
    - [Model trials](#model-trials)
  - [Hardening the code: turning notebook steps into modules](#hardening-the-code-turning-notebook-steps-into-modules)
  - [Serving layer: a simple Flask app](#serving-layer-a-simple-flask-app)
  - [CI/CD & cloud deployment on GCP (Jenkins + Docker + Cloud Run)](#cicd--cloud-deployment-on-gcp-jenkins--docker--cloud-run)
    - [Why not train during `docker build`?](#why-not-train-during-docker-build)
    - [Dockerfile (runtime-only image)](#dockerfile-runtime-only-image)
    - [Jenkins pipeline (Option A: train first, then build)](#jenkins-pipeline-option-a-train-first-then-build)
    - [Secrets & configuration](#secrets--configuration)
  - [Results & trade-offs](#results--trade-offs)
  - [What I’d improve next](#what-id-improve-next)
  - [Run it yourself (dev)](#run-it-yourself-dev)
- [Setup and Installation (Helpful Instructions)](#setup-and-installation-helpful-instructions)
- [Google cloud Setup](#google-cloud-setup)
- [CI/CD Steps using Jenkins, Docker and GCP](#cicd-steps-using-jenkins-docker-and-gcp)
  - [Setup Jenkins Container](#setup-jenkins-container)
  - [Github Integration](#github-integration)
  - [Dockerization of the project](#dockerization-of-the-project)
  - [Create a virtual environment inside the Jenkins container](#create-a-virtual-environment-inside-the-jenkins-container)
  - [Install Google Cloud CLI in Jenkins Container](#install-google-cloud-cli-in-jenkins-container)
  - [Grant Docker Permissions to Jenkins User](#grant-docker-permissions-to-jenkins-user)
  - [Build Docker image of the project](#build-docker-image-of-the-project)
  - [Extract and Push](#extract-and-push)
- [Project Change Logs](#project-change-logs)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## The experimentation notebook: from EDA to model choice

I started in a Jupyter notebook (`experimentation.ipynb`) to quickly iterate.

### EDA highlights

* **Target balance:** Checked cancellation distribution to understand class imbalance.
* **Data cleaning:** Removed duplicate rows; dropped `Booking_ID` and `Unnamed: 0`.
* **Categoricals & numerics:** Reviewed distributions for features like `market_segment_type`, `type_of_meal_plan`, `room_type_reserved`; examined skew for `lead_time` and `avg_price_per_room`.
* **Leakage scan:** Ensured no post-booking signals leak into training.

---
#### Univariate Analysis Plots
![Univariate Analysis Plots](resources/dist_plot.png)

---

#### Bivariate Analysis Plots
![Bivariate Analysis Plots](resources/bivariate_boxplots.png)

---

#### Correlation Plot
![Correlation Plot](resources/corr.png)

---

### Feature engineering & preprocessing (prototyped)

* **Label encoding** for categorical columns (kept mappings).
* **Skewness handling**: `log1p` for skewed numeric columns above a threshold.
* **SMOTE**: Balanced the training set when needed.
* **Feature selection**: Trained a quick Random Forest to get importances, then selected **top-K** features (configurable).

---
#### Imbalanced Dataset
![Imbalanced Dataset before SMOTE](resources/imsmote.png)

---

#### Balanced Dataset after SMOTE
![Balanced Dataset after SMOTE](resources/smote.png)

---

#### Cumulative Feature Importance Curve
![Cumulative Feature Importance Curve](resources/cum_fimp.png)

---

### Model trials

* Baselines: Logistic Regression, Random Forest, XGBoost, LightGBM.
* Metrics: **Accuracy** (primary), plus **Precision/Recall/F1**.
* Results: **Random Forest** topped accuracy but yielded a **\~168 MB** model. **LightGBM** was nearly as accurate but much smaller; this trade-off drove the deployment decision.

#### Performance Comparison
![Performance comparison](resources/model_performance_comparison.png)

---

## Hardening the code: turning notebook steps into modules

After validating the approach in the notebook, I ported the logic into a clean, testable package with config-driven behavior and consistent logging.

### Key modules

* **`src/logger.py`** – Centralized logging (file + console), sensible formats/levels.
* **`src/custom_exception.py`** – Exceptions with file/line context and original error chaining.
* **`utils/utility_functions.py`** – Helpers to read YAML config and load CSV robustly.
* **`src/data_ingestion.py`**

  * Downloads the raw CSV from **GCS** (bucket + blob from `config/config.yaml`) using Application Default Credentials.
  * Splits train/test by ratio; writes to `data/raw/…`.
* **`src/data_preprocessing.py`**

  * Drops unneeded columns, deduplicates.
  * Label-encodes configured categoricals; logs mappings for traceability.
  * Applies log1p to skewed numerics above threshold.
  * Balances with SMOTE (train set).
  * Performs feature selection with RF importances; keeps **top-K** + target.
  * Saves processed train/test to constants: `PROCESSED_TRAIN_DATA_PATH`, `PROCESSED_TEST_DATA_PATH`.
* **`src/model_training.py`**

  * Loads processed data, splits features/target.
  * **LightGBM** tuned via `RandomizedSearchCV` (configurable params).
  * Computes Accuracy/Precision/Recall/F1 (binary-safe with `zero_division=0`).
  * Saves the best model (joblib) to `MODEL_OUTPUT_PATH`.
  * Logs datasets, params, and metrics to **MLflow**.
* **`pipeline/training_pipeline.py`**

  * Orchestrates: **Ingestion → Preprocessing → Training**
  * One function call `run_pipeline()` runs the end-to-end process with clear stage logs and robust error handling.

---

## Serving layer: a simple Flask app

The app is intentionally straightforward for portability and clarity.

* **`application.py`** loads the trained joblib model from `MODEL_OUTPUT_PATH`.
* **`templates/index.html`** + **`static/style.css`** provide a small form to enter the 10 features used in training:

  * `lead_time`, `no_of_special_request`, `avg_price_per_room`, `arrival_month`, `arrival_date`, `market_segment_type`, `no_of_week_nights`, `no_of_weekend_nights`, `type_of_meal_plan`, `room_type_reserved`
* On POST, the app constructs a feature vector in the **exact order** used during training and returns a cancellation prediction (cancel / not cancel).

![Hotel Reservation Cancellation Prediction](resources/app.png)

---

## CI/CD & cloud deployment on GCP (Jenkins + Docker + Cloud Run)

### Why not train during `docker build`?

Running the training step inside `docker build` forces credentials into an image layer and complicates `google-auth` defaults. It also made builds flaky. I moved training **out** of the Dockerfile and into the Jenkins pipeline (with properly scoped credentials), then baked the resulting model artifact into the runtime image.

### Dockerfile (runtime-only image)

* Based on `python:slim`
* Installs system deps (e.g., `libgomp1` for LightGBM)
* Copies the repo and installs the package
* **Does not train**—just runs `application.py` on port **8080**

### Jenkins pipeline (Option A: train first, then build)

Stages:

1. **Clone repo**
2. **Create venv & install**: `pip install -e .`
3. **Train model (with ADC)**:

   * Jenkins injects the GCP service account file as a credential (`withCredentials(file: ...)`).
   * Runs `pipeline/training_pipeline.py` which downloads data from GCS, preprocesses, and trains LightGBM.
   * The model is saved under the repo at **`MODEL_OUTPUT_PATH`** so it gets included by `COPY . .` later.
4. **Build & push image**:

   * Tags with both commit SHA and `latest`
   * Pushes to **GCR** (`gcr.io/<project>/ml-project`)
5. **Deploy to Cloud Run**:

   * `gcloud run deploy ml-project --image gcr.io/<project>/ml-project:<sha> --region us-central1 --platform managed --port 8080 --allow-unauthenticated`

### Secrets & configuration

* **ADC** only during training in Jenkins; never copied into the image.
* The app reads the model from `MODEL_OUTPUT_PATH` at runtime; no cloud credentials are required for serving.
* A `.dockerignore` keeps images lean (`venv/`, `.git/`, caches, local artifacts).

---

## Results & trade-offs

* **Best offline model:** Random Forest (highest accuracy), but **\~168 MB**.
* **Deployed model:** LightGBM (near-parity accuracy), **significantly smaller** binary.
* **Operational benefits:** Faster container pulls, quicker cold starts on Cloud Run, and lower memory footprint → **lower cost** and better UX.

---

## What I’d improve next

* **Persist and load label mappings** so the UI can submit human-readable values and the server maps them to model codes robustly.
* **Add AUC/PR-AUC** for a fuller performance picture.
* **MLflow model registry** + staged promotions (Staging → Production).
* **Monitoring & retraining triggers** (Cloud Run logs + periodic data drift checks).
* **Traffic-split canaries** on Cloud Run for safe rollouts.

---

## Run it yourself (dev)

```bash
# train locally (needs GCP ADC only for data ingestion)
python -m venv .venv && source .venv/bin/activate
pip install -e .
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
python pipeline/training_pipeline.py

# serve locally
python application.py  # http://localhost:8080
```

---

# Setup and Installation (Helpful Instructions)

```bash
# Clone the git repository
git clone https://github.com/SurajBhar/hrprediction.git

# Create a virtual environment and activate it
$ python -m venv /path/to/new/virtual/environment
# To activate the virtual environment in bash/zsh
$ source <venv>/bin/activate
# Virtual Environment using python 
python -m venv hrp
source hrp/bin/activate

# Virtual Environment Using conda (Opt Anyone)
conda create --name hrp python=3.13.0 -y
conda activate hrp

# To install the requirements in the virtual environment
pip install -r requirements.txt

# Alternatively, run setup.py automatically by executing:
pip install -e .

```

# Google cloud Setup
- Create a Google cloud Account with your gmail.
- Activate your free 300 usd credits.
- Install Google Cloud CLI locally on your machine.
- Follow the official instructuions: [MacOs-Install Google cloud CLI](https://cloud.google.com/sdk/docs/install)
- Check your installation: 
    ```bash
        gcloud --version

        # Example Output:
        Google Cloud SDK 532.0.0
        bq 2.1.21
        core 2025.07.25
        gcloud-crc32c 1.0.0
        gsutil 5.35
        
    ```
- Create a Service Account with name: hrpred
- Grant this service account access to hotel-reservation-prediction so that it has permission to complete specific actions on the resources in your project.
- Grant Permissions: 
    - Role: 
        - Strorage Admin: Grants full control of buckets and objects. 
        - Storage Object Viewer: Grants access to view objects and their metadata, excluding ACLs. Can also list the objects in a bucket.

- Go to your buckets
- Edit Access to your bucket >
    - Add Principals> Service Account we just created
    - Assign Roles> Storage Admin, Storage Object Viewer

- Add Key to Your Service Account
    - Go to Service account
    - Click on Actions > Click on Manage Keys > Add Key > Create new key > Json File
    - It will automatically download the Key in a. JSON file to your local machine.

- Export the path to the Key
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/your/credentials.json"

```

---

# CI/CD Steps using Jenkins, Docker and GCP
### Setup Jenkins Container
- Docker in Docker (DID)
    - We will setup a docker container for Jenkins.
    - Inside Jenkins container we will also create one more container for running this project.
    - The inside container is also a docker container. 
    - That is why this is a docker in docker container case.

```bash
    cd custom_jenkins
    # Optional Step
    docker login
    # docker build -t <name-of-docker-container> .
    docker build -t jenkins-dind .
    # Check whether the docker image is listed or not
    docker images
    # To run the docker image
    docker run -d --name jenkins-dind ^
    --privileged ^ # Run in privileged mode to avoid any restrictions
    -p 8080:8080 -p 50000:50000 ^ # To run at 8080 port
    -v //var/run/docker.sock:/var/run/docker.sock ^ # Setup connection between Docker container and jenkins
    -v jenkins_home:/var/jenkins_home ^ # Volume directory for Jenkins, where all the data from jenkins will be stored
    jenkins-dind # Container name

    # Full command:
    docker run -d --name jenkins-dind --privileged -p 8080:8080 -p 50000:50000 -v //var/run/docker.sock:/var/run/docker.sock -v jenkins_home:/var/jenkins_home jenkins-dind
    # Expected output is: Alphanumeric key -> Indicates successful container building
    # Check Running Containers
    docker ps
    # Get Jenkins Logs
    docker logs jenkins-dind

    # Access Jenkins at 8080 port for installation
    localhost:8080

    # To open Jenkins bash terminal
    docker exec -u root -it jenkins-dind bash

    # Install python and pip
    apt update -y # Update all packages and dependencies
    apt install -y python3 # Install python on jenkins container
    python3 --version
    ln -s /usr/bin/python3 /usr/bin/python # Nickname for python3 as python
    python --version
    apt install -y python3-pip # Install pip
    apt install -y python3-venv # Install venv
    exit # Exit Jenkins bash terminal

    # Restart Jenkins Container
    docker restart jenkins-dind

```

---

### Github Integration:
- We will extract the code from the github repository.
- Generate the github access token.
- Connect the github repo to the jenkins project item/workspace.
- Add a Jenkins file to the project.
- Generate pipeline script for the project.
- Add this script to the Jenkins file.
- Test the build inside Jenkins dashboard.
- Check the Console output for success/ failue of build.
- Check the Workspace for the copied github repository.

---

### Dockerization of the project
- Dockerfile to dockerize whole project.

### Create a virtual environment inside the Jenkins container
- This virtual environment will be inside the Jenkins pipeline.

---

### Install Google Cloud CLI in Jenkins Container
Follow these commands to install the Google Cloud SDK inside the Jenkins container:

```bash
    docker exec -u root -it jenkins-dind bash
    apt-get update
    apt-get install -y curl apt-transport-https ca-certificates gnupg
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update && apt-get install -y google-cloud-sdk
    gcloud --version
    exit
```

---

### Grant Docker Permissions to Jenkins User

1. **Grant Docker Permissions:**
Run the following commands to give Docker permissions to the Jenkins user:

    ```bash
    docker exec -u root -it jenkins-dind bash
    groupadd docker
    usermod -aG docker jenkins
    usermod -aG root jenkins
    exit
    ```

2. **Restart Jenkins Container:**
Restart the Jenkins container to apply changes.

    ```bash
    docker restart jenkins-dind
    ```

3. **Enable following API's in GCP**
- Google Container Registry API
- Artifact Registry API
- Cloud Resorce Manager API

---

### Build Docker image of the project
- Here we will utilise the Dockerfile.
- Build Docker Image for the Project
    ```bash
    docker build -t hrprediction_image .
    ```
- Run the Project Docker Container
    ```bash
    docker run -d -p 5000:5000 hrprediction_image
    ```

- Push the image to GCR (Google Cloud Registry).

### Extract and Push
- Extract the image from GCR and push to Google Cloud Run.
- Application deployment is complete.

---

# Project Change Logs
- Blank Structure Created.
- Logging and Exception Implemented.
- Logging and Exception Testing complete.
- Created GCP Setup and Generated JSON Credentials.
- Implemented the Configurations related to GCP.
- Implemented Path Configurations module.
- Implemented utility functions module.
- Implemented Data Ingestion module.
- Performed Data Ingestion.
- Notebook - EDA Complete.
- Notebook - Random Forest Classifier Hyperparameter Tuning and Training
- Notebook - Random Forest Classifier Model Saved
- Notebook - Random Forest Classifier Model Size is approx 168 MB
- Notebook - Will go further with lightgbm model (Smaller in Size)
- Updated configurations
- Implemented Data Preprocessing module.
- Implemented Model Training and MLflow Experiment Tracking.
- Implemented Pipeline by combining data ingestion, preprocessing, tuning, training and tracking.
- Pipeline Automation Verified.
- Flask API/ application build.
- Flask application tested.
- CI/CD Process Workflow Complete
- Updates in Jenkins file.
- Implemented Dockerfile for the project.
- Deployed the Flask app.
- Tested the app on Cloud Run.

---

# Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/XYZ`
3. Make changes & add tests
4. Commit & push: `git push origin feature/XYZ`
5. Open a Pull Request

---

# Support

For questions or issues, please open an [issue](https://github.com/SurajBhar/hrprediction/issues) or write a message to me on [Linkedin](https://www.linkedin.com/in/bhardwaj-suraj/).

If you want to fully replicate this project or want to extend it don't hesitate to contact me. I will be more than happy to provide you with my settings for the deployment.

---

# License
MIT License

Copyright (c) 2025 Suraj Bhardwaj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
