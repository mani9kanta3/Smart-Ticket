# SmartTicket Cloud Deployment
FROM python:3.12-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    python-multipart \
    scikit-learn \
    pandas \
    numpy \
    joblib \
    pyyaml

# Copy only what's needed
COPY src/api/ ./src/api/
COPY configs/ ./configs/

# These will be copied during cloud build
COPY models/tfidf_vectorizer.joblib ./models/
COPY models/baseline_category_model.joblib ./models/
COPY models/baseline_priority_model.joblib ./models/
COPY data/processed/label_mappings.json ./data/processed/

EXPOSE 8080

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]