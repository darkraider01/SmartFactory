# Quick Start Guide

## For First-Time Users

### Step 1: Setup

```bash
cd /app/predictive_maintenance

# Install dependencies (if not already installed)
pip install -r requirements_ml.txt
```

### Step 2: Train Models (Already Done!)

Models are already trained and ready to use:
- ✅ Random Forest Classifier (13.38 MB) - Health Status Prediction
- ✅ XGBoost Classifier (1.66 MB) - Enhanced Fault Detection  
- ✅ LSTM Regressor (0.14 MB) - RUL Prediction

### Step 3: Launch Dashboard

```bash
# Option 1: Direct launch
streamlit run app.py

# Option 2: Use startup script
bash start_app.sh

# Option 3: Background process
nohup streamlit run app.py --server.port 8502 --server.headless true &
```

### Step 4: Access the Application

🌐 **URL**: http://localhost:8502

The dashboard opens automatically in your default browser.

---

## Dashboard Navigation

### 📊 Dashboard Page

**Purpose**: System overview and fleet monitoring

**Features**:
- Fleet health metrics (Total, Healthy, Warning, Critical machines)
- Real-time sensor readings visualization
- Active alerts for machines requiring attention
- Multi-sensor monitoring charts

**What You See**:
- Summary cards showing machine counts
- Line charts with recent sensor data
- Alert boxes highlighting critical/warning units

---

### 🔮 Prediction Page

**Purpose**: Upload data and get AI-powered predictions

#### Using Your Own Data

1. **Prepare CSV File**
   - Required columns: `s1` to `s21` (sensors), `os1` to `os3` (operational settings)
   - Optional: `unit_id`, `cycle`
   - Example format:
   ```csv
   unit_id,cycle,os1,os2,os3,s1,s2,s3,...,s21
   1,1,-0.0007,0.0002,100.0,518.67,641.82,1589.70,...,23.4190
   ```

2. **Upload File**
   - Click "Upload CSV file with sensor readings"
   - Browse and select your file
   - System validates and shows preview

3. **Visualize Sensors**
   - Select sensors from dropdown (e.g., s1, s2, s3)
   - Interactive Plotly chart shows trends
   - Zoom, pan, and hover for details

4. **Generate Predictions**
   - Click "🔮 Generate Predictions" button
   - AI models process your data
   - Results appear within seconds

#### Using Sample Data

1. Click "📝 Use Sample Data" button
2. System generates synthetic sensor data automatically
3. Data appears in preview table
4. Proceed to generate predictions

#### Understanding Results

**1. Key Metrics**
- **Predicted RUL**: Remaining operational cycles before failure
- **Health Status**: Healthy (>100 cycles) / Warning (50-100) / Critical (<50)
- **Confidence**: Model certainty percentage

**2. Status Indicator**
- 🟢 **Healthy**: Normal operation, routine maintenance
- 🟡 **Warning**: Early degradation signs, schedule inspection
- 🔴 **Critical**: Imminent failure, immediate action required

**3. Model Comparison**
- Shows predictions from multiple models
- Compare Random Forest vs XGBoost classifications
- LSTM provides RUL regression

**4. Maintenance Recommendations**
- ✅ Healthy: Continue monitoring, schedule routine inspection
- ⚠️ Warning: Perform detailed inspection, order spare parts
- 🚨 Critical: Stop operation, schedule immediate maintenance

---

### 📈 Analytics Page

**Purpose**: Historical trends and fleet insights

**Features**:
- Fleet health distribution (pie chart)
- RUL distribution across all machines (bar chart)
- 30-day maintenance activity timeline
- Failure and maintenance event tracking

**Use Cases**:
- Identify patterns in machine degradation
- Optimize maintenance scheduling
- Resource planning based on predictions

---

### ℹ️ About Page

**Purpose**: System documentation and technical details

**Information**:
- Technology stack and architecture
- Model performance metrics
- Dataset information (NASA C-MAPSS)
- Getting started instructions
- Links to documentation

---

## Example Workflows

### Workflow 1: Daily Health Check

1. Open Dashboard page
2. Review fleet metrics
3. Check active alerts
4. Investigate any critical/warning machines
5. Export alerts for maintenance team

### Workflow 2: Predictive Analysis

1. Go to Prediction page
2. Upload latest sensor readings CSV
3. Visualize key sensors (temperature, vibration, pressure)
4. Generate predictions
5. Review recommendations
6. Create maintenance work orders for critical machines

### Workflow 3: Trend Analysis

1. Navigate to Analytics page
2. Review fleet health distribution
3. Analyze RUL distribution patterns
4. Check 30-day maintenance history
5. Identify recurring issues
6. Adjust preventive maintenance schedules

---

## Tips & Best Practices

### For Best Predictions

✅ **Do**:
- Upload at least 50 timesteps of sensor data per machine
- Ensure all 21 sensors have valid readings
- Include operational settings (os1, os2, os3)
- Use consistent sampling rates
- Normalize sensor values if needed

❌ **Don't**:
- Upload incomplete or corrupted data
- Mix data from different machine types
- Use data with large gaps in time series
- Ignore warning messages

### Data Quality

- **Missing Values**: System handles some missing data, but complete records are better
- **Outliers**: Extreme sensor values may affect predictions
- **Frequency**: More frequent sampling improves LSTM predictions

### Model Selection

- **Random Forest**: Fast, reliable for general classification
- **XGBoost**: Best accuracy for fault type prediction
- **LSTM**: Superior for time-series RUL estimation

---

## Sample Data Format

### Minimal CSV (3 timesteps)

```csv
unit_id,cycle,os1,os2,os3,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21
1,1,-0.0007,0.0002,100,0.52,0.63,0.71,0.58,0.49,0.61,0.55,0.47,0.53,0.59,0.48,0.62,0.51,0.57,0.46,0.60,0.54,0.50,0.56,0.52
1,2,-0.0004,-0.0003,100,0.53,0.64,0.72,0.59,0.50,0.62,0.56,0.48,0.54,0.60,0.49,0.63,0.52,0.58,0.47,0.61,0.55,0.51,0.57,0.53
1,3,0.0001,0.0001,100,0.54,0.65,0.73,0.60,0.51,0.63,0.57,0.49,0.55,0.61,0.50,0.64,0.53,0.59,0.48,0.62,0.56,0.52,0.58,0.54
```

### Download Sample

Generate sample data in Python:

```python
import pandas as pd
import numpy as np

# Generate 50 timesteps
data = {
    'unit_id': [1] * 50,
    'cycle': range(1, 51),
    'os1': np.random.uniform(-0.5, 0.5, 50),
    'os2': np.random.uniform(-0.5, 0.5, 50),
    'os3': np.ones(50) * 100
}

# Add 21 sensors
for i in range(1, 22):
    base = np.random.uniform(0.4, 0.6)
    trend = np.linspace(0, 0.2, 50)
    noise = np.random.normal(0, 0.05, 50)
    data[f's{i}'] = base + trend + noise

df = pd.DataFrame(data)
df.to_csv('my_sensor_data.csv', index=False)
print("Sample data saved to my_sensor_data.csv")
```

---

## Troubleshooting

### Issue: Dashboard won't load

**Solution**:
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Check logs
tail -f streamlit.log

# Restart
pkill streamlit
streamlit run app.py
```

### Issue: Prediction fails

**Causes**:
- Incorrect CSV format
- Missing required columns
- Insufficient data (need 50+ rows)

**Solution**:
- Verify CSV has columns s1-s21, os1-os3
- Check for NaN values
- Use sample data to test

### Issue: Models not loaded

**Solution**:
```bash
# Check models directory
ls -lh models/

# Retrain if needed
python train_lightweight.py

# Restart app
pkill streamlit
streamlit run app.py
```

### Issue: Slow predictions

**Causes**:
- Large CSV files
- Limited system resources

**Solution**:
- Reduce data to last 100 timesteps
- Close other applications
- Use lightweight models

---

## Advanced Usage

### Batch Processing

```python
from data_preprocessing import DataPreprocessor
from model_training import RandomForestModel
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Process multiple files
files = ['machine1.csv', 'machine2.csv', 'machine3.csv']
for file in files:
    # Load and predict
    # ... your processing code
```

### Custom Thresholds

Edit `config.yaml`:

```yaml
thresholds:
  healthy: 120    # Increase from 100
  warning: 60     # Increase from 50
  critical: 60    # RUL <= 60 is critical
```

### API Integration

See `DEPLOYMENT.md` for FastAPI wrapper example.

---

## Next Steps

1. ✅ **Explore Dashboard**: Familiarize yourself with all pages
2. ✅ **Test Predictions**: Upload sample or real data
3. ✅ **Analyze Results**: Understand health status and RUL
4. ✅ **Integrate**: Connect with your maintenance system
5. ✅ **Monitor**: Track prediction accuracy over time
6. ✅ **Retrain**: Update models with new failure data

---

## Getting Help

📖 **Documentation**:
- `README.md` - Full project documentation
- `DEPLOYMENT.md` - Production deployment guide
- `config.yaml` - Configuration options

🧪 **Testing**:
```bash
python test_system.py  # Verify installation
```

🔧 **Support**:
- Check logs: `tail -f streamlit.log`
- Review configuration: `cat config.yaml`
- Test models: `python test_system.py`

---

## Quick Reference

| Task | Command |
|------|----------|
| Start Dashboard | `streamlit run app.py` |
| Train Models | `python train_lightweight.py` |
| Test System | `python test_system.py` |
| Check Logs | `tail -f streamlit.log` |
| Stop App | `pkill streamlit` |
| View Models | `ls -lh models/` |
| View Data | `ls -lh data/` |

---

**Ready to start? Launch the dashboard and explore!** 🚀

```bash
streamlit run app.py
```