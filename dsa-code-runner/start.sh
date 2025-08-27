#!/bin/bash
# This script first builds all Docker images and then starts the Flask web server.
# It's the all-in-one command to launch the project.
# chmod +x start.sh && ./start.sh
# Set echo on and exit on first error
set -e

echo "--- Step 1: Building Docker Images ---"
# Call the existing build script to handle the Docker builds
./build.sh

echo ""
echo "--- Step 2: Starting the Flask Server ---"
echo "Server will be available at http://localhost:3000"
echo "Press CTRL+C to stop the server."

# Run the Python application
cd ..
source .venv/bin/activate
cd dsa-code-runner
python3 app.py

echo ""
echo "--- Server stopped. ---"
