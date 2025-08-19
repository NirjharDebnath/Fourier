# Use a lightweight base image like Debian Slim
FROM debian:bullseye-slim

# Install the C++ compiler and build essentials
# -y automatically confirms the installation
RUN apt-get update && apt-get install -y g++ build-essential

# Set the working directory inside the container
WORKDIR /usr/src/app

# Default command (will be overridden)
CMD ["/bin/bash"]
