# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code to the container
COPY . .

# Define environment variable for Flask (if using Flask as web server)
ENV FLASK_APP=score.py

# Command to run your entry script
CMD ["python", "score.py"]
