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

