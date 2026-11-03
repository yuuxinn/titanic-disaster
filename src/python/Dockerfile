# Python container for the Titanic logistic regression model
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script (the data is NOT copied - it is mounted at run time)
COPY main.py .

# Print output immediately instead of buffering
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
