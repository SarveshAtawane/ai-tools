# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all source code
COPY . .
COPY . /app/

# Expose port for the server
EXPOSE 8000

# Command to run the server
CMD ["hypercorn", "--bind", "0.0.0.0:8000", "api:app"]