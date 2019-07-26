pipeline {
  agent {
    docker {
      image 'tensorflow/tensorflow:1.13.1-gpu-py3'
    }

  }
  stages {
    stage('setup') {
      steps {
        sh 'pwd'
        sh '''virtualenv .venv
source .venv/bin/activate
cd pyspec
pip3 install -r requirements.txt'''
      }
    }
    stage('test') {
      steps {
        sh '''source .venv/bin/activate
cd pyspec
pytest ./'''
      }
    }
  }
}