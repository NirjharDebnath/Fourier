#!/bin/bash
# This script builds all the necessary Docker images for the DSA Code Runner.
# To run it, first make it executable with: chmod +x build.sh
# Then, execute it with: ./build.sh

# Set echo on and exit on first error
set -e

echo "--- Building Python runner image (python-runner) ---"
docker build -t python-runner -f Dockerfile .

echo ""
echo "--- Building C/C++ runner image (cpp-runner) ---"
docker build -t cpp-runner -f Dockerfile.cpp .

echo ""
echo "--- Building Java runner image (java-runner) ---"
docker build -t java-runner -f Dockerfile.java .

echo ""
echo "✅ All Docker images built successfully!"
