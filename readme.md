# Ping Response Time Collector + Deep Learning Latency Predictor

This project collects real-time ping latency data for domains like Google, Amazon, and OpenAI, and uses a deep learning model to predict future network latency based on additional simulated features.

## 🔧 Features
- Collects real network latency using `ping`
- Saves data to `ping_latency.csv`
- Adds simulated network load and packet size
- Scales data and trains a regression model (TensorFlow/Keras)
- Visualizes actual vs predicted latency in a scatter plot

## 📁 Files
- `full_pipeline.py` – Main script that does everything (ping + DL + plot)
- `ping_latency.csv` – Generated CSV with latency data (added to `.gitignore`)
- `README.md` – This file

## 📦 Requirements
- Python 3.7+
- Packages: `pandas`, `numpy`, `matplotlib`, `tensorflow`, `scikit-learn`

Install with:
```bash
pip install -r requirements.txt
