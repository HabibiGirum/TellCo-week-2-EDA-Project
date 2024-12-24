# Base Image
FROM python:3.11

# Set Work Directory
WORKDIR /app

# Copy Requirements
COPY requirements.txt .

# Install Dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy Source Code
COPY . .

# Expose Port (Assuming Streamlit dashboard runs on 8501)
EXPOSE 8501

# Set Default Command to Run Streamlit Dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
