FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rfm/ rfm/
COPY pipeline.py app.py ./
COPY Dataset/ Dataset/

# Pre-compute results at build time so the dashboard starts instantly.
RUN python pipeline.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
