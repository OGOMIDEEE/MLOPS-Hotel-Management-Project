pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage("Cloning Github repo to Jenkins"){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins........'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/OGOMIDEEE/MLOPS-Hotel-Management-Project.git']])
                }
            }
        }

        stage("Setting up our virtual Environment and installing dependencies"){
            steps{
                script{
                    echo 'Setting up our virtual Environment and installing dependencies........'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                    
                }
            }
        }
    }
}