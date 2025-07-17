import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

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

# Load data and train model
@st.cache_resource
def load_data_and_train_model():
    try:
        df = pd.read_csv("deliverytime.txt")
        df['distance'] = df.apply(lambda row: calculate_distance(
            row['Restaurant_latitude'],
            row['Restaurant_longitude'],
            row['Delivery_location_latitude'],
            row['Delivery_location_longitude']
        ), axis=1)

        X = df[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]]
        y = df["Time_taken(min)"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return model, True
    except Exception as e:
        st.warning(f"Using simplified prediction method. Model loading failed: {str(e)}")
        return None, False

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
    
    if not model_loaded:
        st.info("Note: Using simplified prediction method as the machine learning model couldn't be loaded")
    
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app predicts food delivery time based on:
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
            
            # Make prediction
            with st.spinner('Calculating delivery time...'):
                try:
                    if model_loaded:
                        features = np.array([[age, ratings, distance]])
                        prediction = model.predict(features)[0]
                    else:
                        # Fallback formula if model isn't available
                        prediction = max(15, min(120, 
                                            distance * 2 + 
                                            (5 - ratings) * 5 + 
                                            (35 - age) * 0.5))
                    
                    time.sleep(1)  # Just for effect
                    
                    st.success("Prediction complete!")
                    st.subheader("Predicted Delivery Time")
                    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{int(prediction)} minutes</h1>", 
                                unsafe_allow_html=True)
                    
                    # Visual indicator
                    if prediction < 30:
                        st.success("Fast delivery expected! 🚀")
                    elif prediction < 45:
                        st.info("Average delivery time ⏱️")
                    else:
                        st.warning("Longer delivery time expected 🐢")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        else:
            st.error("Please enter valid location coordinates for both restaurant and delivery location.")

if __name__ == "__main__":
    main()
