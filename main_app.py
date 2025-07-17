import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import time

# Load data
df = pd.read_csv('https://raw.github.com/Vasanthkumar5648/Food-Delivery-Time-Prediction/main/deliverytime.txt')
df.head()
df.info()
df.isnull().sum()

# Constants
R = 6371  # Earth's radius in km

# Function to convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Haversine formula to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Calculate distances for the dataset
df['distance'] = df.apply(lambda row: calculate_distance(
    row['Restaurant_latitude'],
    row['Restaurant_longitude'],
    row['Delivery_location_latitude'],
    row['Delivery_location_longitude']
), axis=1)

# Prepare features and target
x = np.array(df[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]])
y = np.array(df[["Time_taken(min)"]])

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Build and train model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)

# Streamlit app
def main():
    st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🍔", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #ffffff;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🍔 Food Delivery Time Prediction")
    st.markdown("Predict how long your food delivery will take based on delivery partner details and distance.")
    
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app predicts food delivery time using an LSTM neural network model. 
        The prediction is based on:
        - Delivery partner's age
        - Delivery partner's ratings
        - Distance between restaurant and delivery location
        """)    
    
    # Input form
    with st.form("delivery_form"):
        st.header("Enter Delivery Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age of Delivery Partner", min_value=18, max_value=70, value=25)
            ratings = st.slider("Ratings of Delivery Partner", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        
        with col2:
            st.subheader("Restaurant Location")
            rest_lat = st.number_input("Restaurant Latitude", format="%.6f")
            rest_lon = st.number_input("Restaurant Longitude", format="%.6f")
            
            st.subheader("Delivery Location")
            deliv_lat = st.number_input("Delivery Location Latitude", format="%.6f")
            deliv_lon = st.number_input("Delivery Location Longitude", format="%.6f")
        
        submitted = st.form_submit_button("Predict Delivery Time")
    
    # When form is submitted
    if submitted:
        if rest_lat and rest_lon and deliv_lat and deliv_lon:
            # Calculate distance
            distance = calculate_distance(rest_lat, rest_lon, deliv_lat, deliv_lon)
            
            # Display inputs
            st.subheader("Input Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Delivery Partner Age", f"{age} years")
            col2.metric("Delivery Partner Ratings", f"{ratings}/5")
            col3.metric("Distance", f"{distance:.2f} km")
            
            # Prepare input for prediction
            features = np.array([[age, ratings, distance]]).reshape(1, 3, 1)
            
            # Make prediction
            with st.spinner('Calculating delivery time...'):
                prediction = model.predict(features)[0][0]
                time.sleep(1)  # Just for effect
                
                st.success("Prediction complete!")
                st.subheader("Predicted Delivery Time")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{int(prediction)} minutes</h1>", unsafe_allow_html=True)
                
                # Visual indicator
                if prediction < 30:
                    st.success("Fast delivery expected! 🚀")
                elif prediction < 45:
                    st.info("Average delivery time ⏱️")
                else:
                    st.warning("Longer delivery time expected 🐢")
        else:
            st.error("Please enter valid location coordinates for both restaurant and delivery location.")

if __name__ == "__main__":
    main()
