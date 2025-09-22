#!/bin/bash
# Project Sentinel Setup Script
# This script sets up the development environment for the AI-powered UFDR analysis platform

echo "🚀 Setting up Project Sentinel - AI-Powered UFDR Analysis Platform"
echo "================================================================="

# Check if Python 3.8+ is installed
echo "📋 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "📥 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created. Please edit it with your API keys and configuration."
else
    echo "ℹ️ .env file already exists."
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/sample data/vector_db

# Set up Git hooks (if .git exists)
if [ -d .git ]; then
    echo "🔧 Setting up Git hooks..."
    # Add pre-commit hook for code formatting
    cat > .git/hooks/pre-commit << EOF
#!/bin/bash
# Run black formatter
black src/ --check
if [ \$? -ne 0 ]; then
    echo "❌ Code formatting issues found. Run 'black src/' to fix them."
    exit 1
fi

# Run flake8 linter
flake8 src/
if [ \$? -ne 0 ]; then
    echo "❌ Linting issues found. Please fix them before committing."
    exit 1
fi
EOF
    chmod +x .git/hooks/pre-commit
fi

echo ""
echo "✅ Project Sentinel setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Start Neo4j database (Neo4j Desktop or Docker)"
echo "3. Run the application:"
echo "   - Backend API: uvicorn src.api.main:app --reload"
echo "   - Frontend: streamlit run src/frontend/app.py"
echo ""
echo "📚 For more information, check the documentation in the docs/ folder."
echo ""
echo "🎯 Ready to transform forensic investigation with AI!"