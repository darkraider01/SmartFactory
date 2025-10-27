# Deployment Guide

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements_ml.txt

# 2. Train models (first time only)
python train_lightweight.py

# 3. Start the dashboard
streamlit run app.py

# Or use the startup script
bash start_app.sh
```

### Access the Application

- Local: http://localhost:8502
- The dashboard will open automatically in your browser

## Docker Deployment (Optional)

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy application files
COPY . .

# Create directories
RUN mkdir -p data models results logs

# Train models on first run
RUN python train_lightweight.py

# Expose Streamlit port
EXPOSE 8502

# Start application
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build Docker image
docker build -t predictive-maintenance .

# Run container
docker run -p 8502:8502 predictive-maintenance
```

## Cloud Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Visit https://share.streamlit.io
3. Connect your repository
4. Deploy!

**Note**: Ensure models are trained before deployment or include training in startup.

### AWS EC2

```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@ec2-ip

# Install Python and dependencies
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements_ml.txt

# Train models
python3 train_lightweight.py

# Run with nohup
nohup streamlit run app.py --server.port 8502 &

# Configure security group to allow port 8502
```

### Google Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/predictive-maintenance

# Deploy
gcloud run deploy predictive-maintenance \
  --image gcr.io/PROJECT_ID/predictive-maintenance \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Production Considerations

### Performance Optimization

1. **Model Caching**: Models are cached using `@st.cache_resource`
2. **Data Preprocessing**: Use batch processing for multiple predictions
3. **Memory**: Lightweight models trained with reduced complexity

### Security

1. **Authentication**: Add Streamlit authentication for production
2. **HTTPS**: Use reverse proxy (Nginx) with SSL certificates
3. **Rate Limiting**: Implement API rate limiting for predictions

### Monitoring

1. **Logs**: Check `streamlit.log` for errors
2. **Metrics**: Monitor prediction latency and accuracy
3. **Alerts**: Set up alerts for model drift or failures

### Scaling

1. **Horizontal**: Deploy multiple instances behind load balancer
2. **Vertical**: Use machines with more RAM/CPU for larger datasets
3. **Model Serving**: Consider TensorFlow Serving for high-throughput

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8502
lsof -i :8502

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8503
```

### Models Not Loading

```bash
# Retrain models
python train_lightweight.py

# Check model files
ls -lh models/
```

### Memory Issues

```bash
# Reduce batch size in config.yaml
# Use lightweight training script
python train_lightweight.py
```

## API Integration (Optional)

### FastAPI Wrapper

Create `api.py` for REST API:

```python
from fastapi import FastAPI
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models
rf_model = joblib.load('models/random_forest_model.pkl')
lstm_model = load_model('models/lstm_model.h5')

@app.post("/predict")
async def predict(data: list):
    X = np.array(data).reshape(1, 50, 34)
    X_flat = X.reshape(1, -1)
    
    status = int(rf_model.predict(X_flat)[0])
    rul = float(lstm_model.predict(X, verbose=0)[0][0])
    
    return {"status": status, "rul": rul}
```

Run with: `uvicorn api:app --port 8000`

## Maintenance

### Model Retraining

```bash
# Retrain with new data
python train_models.py

# Or use lightweight version
python train_lightweight.py
```

### Backup

```bash
# Backup models
tar -czf models_backup.tar.gz models/

# Backup data
tar -czf data_backup.tar.gz data/
```

### Updates

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements_ml.txt --upgrade

# Restart application
pkill streamlit
streamlit run app.py
```

## Support

For issues or questions:
- Check logs: `tail -f streamlit.log`
- Review test results: `python test_system.py`
- Consult README.md for detailed documentation
