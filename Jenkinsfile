pipeline {
    agent any

    environment {
        VENV_DIR   = 'venv'
        GCP_PROJECT = 'utility-node-467910-j3'
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
        DOCKER_IMAGE = "gcr.io/${GCP_PROJECT}/ml-project"
        DOCKER_BUILDKIT = '1'
    }

    stages {
        stage('Cloning Github repo to Jenkins') {
            steps {
                script {
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/SurajBhar/hrprediction']]
                    )
                }
            }
        }

        stage('Setting up Virtual Environment & Installing dependencies') {
            steps {
                sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                '''
            }
        }

        stage('Train Model (uses GCS via ADC)') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh '''
                        . ${VENV_DIR}/bin/activate
                        # ADC is picked up automatically by google-auth from GOOGLE_APPLICATION_CREDENTIALS
                        python pipeline/training_pipeline.py
                    '''
                }
            }
        }

        stage('Building and Pushing Docker Image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}
                        gcloud auth configure-docker --quiet

                        # Build once, tag with commit SHA and latest
                        docker build -t ${DOCKER_IMAGE}:${GIT_COMMIT} -t ${DOCKER_IMAGE}:latest .

                        docker push ${DOCKER_IMAGE}:${GIT_COMMIT}
                        docker push ${DOCKER_IMAGE}:latest
                    '''
                }
            }
        }

        stage('Deploy to Google Cloud Run') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project ${GCP_PROJECT}

                        gcloud run deploy ml-project \
                            --image=${DOCKER_IMAGE}:${GIT_COMMIT} \
                            --platform=managed \
                            --region=us-central1 \
                            --port=5000 \
                            --allow-unauthenticated
                    '''
                }
            }
        }
    }
}
