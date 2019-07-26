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
        sh '''cd pyspec
pip3 --user install -r requirements.txt'''
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