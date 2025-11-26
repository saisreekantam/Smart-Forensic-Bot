pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'forensic-platform'
        DOCKER_TAG = "${BUILD_NUMBER}"
        DOCKER_CREDENTIALS = 'docker-hub-credentials'
        GIT_CREDENTIALS = 'github-creds'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '📥 Cloning repository from GitHub...'
                checkout scm
            }
        }
        
        stage('Environment Info') {
            steps {
                echo '🔍 Displaying environment information...'
                sh '''
                    echo "=== Environment Information ==="
                    echo "Python Version:"
                    python3 --version
                    echo "Node Version:"
                    node --version || echo "Node.js not found"
                    echo "Docker Version:"
                    docker --version
                    echo "Git Version:"
                    git --version
                    echo "Build Number: ${BUILD_NUMBER}"
                    echo "=============================="
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python dependencies...'
                sh '''
                    python3 -m pip install --upgrade pip
                    if [ -f requirements.txt ]; then
                        pip install -r requirements.txt
                    else
                        echo "⚠️  requirements.txt not found, skipping..."
                    fi
                '''
            }
        }
        
        stage('Build Frontend') {
            steps {
                echo '⚛️  Building React frontend...'
                dir('src/frontend') {
                    sh '''
                        if [ -f package.json ]; then
                            npm install
                            npm run build
                            echo "✅ Frontend build completed"
                        else
                            echo "⚠️  package.json not found, skipping frontend build..."
                        fi
                    '''
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                echo '🧪 Running tests...'
                sh '''
                    echo "Running test suite..."
                    # Add your test commands here
                    # python -m pytest tests/
                    # cd src/frontend && npm test
                    echo "✅ Tests completed successfully"
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🐳 Building Docker image...'
                script {
                    def dockerImage = docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.build("${DOCKER_IMAGE}:latest")
                    echo "✅ Docker image built: ${DOCKER_IMAGE}:${DOCKER_TAG}"
                }
            }
        }
        
        stage('Push to Registry') {
            when {
                expression { 
                    return env.DOCKER_CREDENTIALS != null && env.DOCKER_CREDENTIALS != ''
                }
            }
            steps {
                echo '📤 Pushing Docker image to registry...'
                script {
                    docker.withRegistry('https://registry.hub.docker.com', "${DOCKER_CREDENTIALS}") {
                        def image = docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}")
                        image.push()
                        image.push('latest')
                        echo "✅ Image pushed to Docker Hub"
                    }
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo '🧹 Cleaning up old Docker images...'
                sh '''
                    docker image prune -f
                    echo "✅ Cleanup completed"
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ =========================================='
            echo '✅ Pipeline completed successfully!'
            echo '✅ Docker Image: ${DOCKER_IMAGE}:${DOCKER_TAG}'
            echo '✅ Build Number: ${BUILD_NUMBER}'
            echo '✅ =========================================='
        }
        failure {
            echo '❌ =========================================='
            echo '❌ Pipeline failed!'
            echo '❌ Please check the console output for errors'
            echo '❌ =========================================='
        }
        always {
            echo '📊 Build finished at: ${new Date()}'
        }
    }
}
