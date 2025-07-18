# 🚚 Food Delivery Time Prediction App

![App Screenshot](./images/app-screenshot.png)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

## 🌟 Overview

A machine learning-powered web app that predicts food delivery time with 85% accuracy using:
- Delivery partner attributes
- Geographic coordinates
- Random Forest regression model

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📍 Precise Distance Calculation | Uses Haversine formula for accurate geospatial distance |
| ⏱️ Real-time Prediction | Instant delivery time estimates |
| 📊 Interactive Dashboard | User-friendly Streamlit interface |
| 🎯 Performance Metrics | Visual indicators for prediction quality |
| 📱 Mobile Responsive | Works across all device types |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/food-delivery-predictor.git
cd food-delivery-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
