#!/bin/bash

# Project Sentinel Startup Script
# Runs both the FastAPI backend and Streamlit frontend

echo "🔍 Starting Project Sentinel - Forensic Investigation Platform"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, fastapi, uvicorn, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Installing..."
    pip3 install streamlit fastapi uvicorn requests pandas
fi

# Create logs directory
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Project Sentinel..."
    kill $API_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "✅ Shutdown complete"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the FastAPI backend
echo "🚀 Starting FastAPI backend on http://localhost:8000..."
python3 demo_case_platform.py > logs/api.log 2>&1 &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to initialize..."
sleep 5

# Check if API is running
if ! ps -p $API_PID > /dev/null; then
    echo "❌ Failed to start API backend. Check logs/api.log for details."
    exit 1
fi

# Test API connection
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null; then
        echo "✅ API backend is ready"
        break
    else
        echo "⏳ Waiting for API to be ready... ($i/10)"
        sleep 2
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ API backend failed to start properly. Check logs/api.log"
        cleanup
        exit 1
    fi
done

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend on http://localhost:8501..."
streamlit run streamlit_demo.py --server.port=8501 --server.headless=true > logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

# Check if Streamlit is running
if ! ps -p $STREAMLIT_PID > /dev/null; then
    echo "❌ Failed to start Streamlit frontend. Check logs/streamlit.log for details."
    cleanup
    exit 1
fi

echo ""
echo "🎉 Project Sentinel is now running!"
echo "============================================================"
echo "📊 Backend API:  http://localhost:8000"
echo "🌐 Web Frontend: http://localhost:8501"
echo "📋 API Docs:     http://localhost:8000/docs"
echo ""
echo "👮‍♂️ For officers: Open http://localhost:8501 in your browser"
echo "🔧 For developers: Check API documentation at http://localhost:8000/docs"
echo ""
echo "📝 Logs are saved in the logs/ directory"
echo "🛑 Press Ctrl+C to stop both services"
echo ""

# Wait for user interruption
while true; do
    sleep 1
    
    # Check if processes are still running
    if ! ps -p $API_PID > /dev/null; then
        echo "❌ API backend stopped unexpectedly. Check logs/api.log"
        cleanup
        exit 1
    fi
    
    if ! ps -p $STREAMLIT_PID > /dev/null; then
        echo "❌ Streamlit frontend stopped unexpectedly. Check logs/streamlit.log"
        cleanup
        exit 1
    fi
done