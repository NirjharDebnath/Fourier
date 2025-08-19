# Use an official lightweight OpenJDK image as the parent image
# We're using Java 17, a modern Long-Term Support (LTS) version
FROM openjdk:17-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Default command (will be overridden by our script)
CMD ["/bin/bash"]
