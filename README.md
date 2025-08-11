# Hotel Reservation Prediction using ML

This is the starting of the project. Building the structure of the project.

# Setup and Installation

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

### Dockerization of the project
- Dockerfile to dockerize whole project.


### Create a virtual environment
- This virtual environment will be inside the Jenkins pipeline.

### Build Docker image of the project
- Here we will utilise the Dockerfile.
- Build the docker image.
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
- 
