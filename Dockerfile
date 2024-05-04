# Use the official Python image as base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Expose the port that Flask runs on
EXPOSE 5000

# Define the command to run the application when the container starts
CMD [ "python", "./app.py" ]
