# Use an official lightweight Python image as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
# We will mount the actual student code file at runtime, so this is optional
# COPY . .

# Command to run the python script
# We will override this command when we run the container
CMD ["python3", "./main.py"]
