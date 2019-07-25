pipeline {
  agent {
    docker {
      image 'nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04'
    }

  }
  stages {
    stage('setup') {
      steps {
        sh 'apt install python3-pip'
        sh '''cd pyspec
pip3 install -r requirements.txt'''
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
