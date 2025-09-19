# Stage 1: Builder - This stage is for installing dependencies, including build-time ones.
FROM python:3.11-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for building Python packages
# --no-install-recommends prevents installing optional packages, reducing size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to keep dependencies isolated
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
# Add the venv to the PATH, so 'pip' and 'python' commands use it
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install Python dependencies into the virtual environment
# --no-cache-dir reduces layer size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image - This is the lean, production-ready image.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Create a non-root user for better security
RUN useradd --create-home appuser
USER appuser

# Copy the virtual environment from the builder stage. This contains all the installed dependencies.
COPY --from=builder --chown=appuser:appuser /app/venv /app/venv

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Add the virtual environment to the PATH for the final image as well
ENV PATH="/app/venv/bin:$PATH"
# Ensure Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application using uvicorn.
# It will use the Python from our virtual environment.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]