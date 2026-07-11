# Base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Show logs immediately
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install .

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]