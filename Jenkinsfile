pipeline {
  agent {
    docker {
      image 'eros.fiehnlab.ucdavis.edu/jenkins-agent:latest'
      args '--device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia1'
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