# Use an official Python runtime as a base image
FROM python:3.9.19

# Set the working directory in the container
WORKDIR /app

# Install system libraries required by numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev

# Copy the local directory files to the container's workspace
COPY . /app

# Install the necessary packages specified in requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4 \
                                 pandas==2.2.2 \
                                 matplotlib==3.3.1 \
                                 seaborn==0.11.0 \
                                 plotly==5.22.0 \
                                 streamlit==1.34.0 \
                                 scikit-learn==1.4.2 \
                                 tensorflow-cpu==2.16.1 \
                                 protobuf==3.20.3 \
                                 altair==4.1.0 \
                                 click==8.0.0

EXPOSE 8501

# Run the Streamlit application command
CMD streamlit run --server.port $PORT app.py
