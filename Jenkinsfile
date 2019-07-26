pipeline {
  agent {
    docker {
      image 'eros.fiehnlab.ucdavis.edu/jenkins-agent:latest'
    }

  }
  stages {
    stage('setup') {
      steps {
        sh 'pwd'
        sh '''virtualenv .venv
source .venv/bin/activate
cd pyspec
pip3 install --user -r requirements.txt'''
      }
    }
    stage('test') {
      steps {
        sh '''cd pyspec
pytest ./'''
      }
    }
  }
}