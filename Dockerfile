# --- STAGE 1: The "Builder" ---
# We use a full Python image here because it has the build tools
# needed to compile some of the Python packages.
FROM python:3.11-bookworm AS builder

WORKDIR /app

# Install system-level build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies into the venv
COPY requirements.txt .

# Upgrade pip in its own step (this is fine)
RUN pip install --no-cache-dir --upgrade pip

# --- THIS IS THE NEW, COMBINED COMMAND ---
# Install torch (CPU) AND all requirements in a single RUN command.
# This creates a single layer and avoids VM crashes between steps.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# --- STAGE 2: The "Final" Image ---
# Now we use the "slim" image, which is much smaller
FROM python:3.11-slim-bookworm AS final

WORKDIR /app

# Copy *only* the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy your application code
COPY . .

# Create a non-root user for better security
RUN mkdir -p /app/db /app/Exports && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set the PATH to use the venv
ENV PATH="/app/venv/bin:$PATH"

# Expose the port your FastAPI app runs on
EXPOSE 8000

# This is the new command to run your FastAPI web server
# 'app:app' means "in file app.py, find the variable named app"
# '0.0.0.0' is required to make it accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
