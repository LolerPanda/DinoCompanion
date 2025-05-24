#!/bin/bash
# Run DinoCompanion demo

echo "🦕 DinoCompanion Demo Launcher"
echo "=============================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run selected demo
echo ""
echo "Select demo to run:"
echo "1) Simple console demo"
echo "2) Streamlit web app"
echo "3) Run tests"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Running simple demo..."
        python examples/simple_demo.py
        ;;
    2)
        echo "Starting Streamlit app..."
        echo "Open http://localhost:8501 in your browser"
        streamlit run app.py
        ;;
    3)
        echo "Running tests..."
        python tests/test_basic.py
        ;;
    *)
        echo "Invalid choice. Please run again and select 1, 2, or 3."
        ;;
esac 