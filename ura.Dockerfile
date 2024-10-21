# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose port 8080 for API
EXPOSE 8080

# Command to run your API application
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
