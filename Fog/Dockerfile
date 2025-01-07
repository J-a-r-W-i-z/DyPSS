# Use an official Python image for ARM as the base image
FROM ultralytics/ultralytics:latest-arm64

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY node.py .

COPY best.pt .

RUN pip install requests
RUN pip install psutil

CMD ["python", "-u", "node.py", "5000"]