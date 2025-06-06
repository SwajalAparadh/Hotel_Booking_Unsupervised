
import os
os.system('pip install joblib')
import joblib


import streamlit as st
import pandas as pd
import joblib


model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')


segment_labels = {
    0: "Budget Traveler",
    1: "Luxury Seeker",
    2: "Family Vacationer",
    3: "Business Traveler"
    # Modify or add more as per your model's actual clusters
}


st.title("🏨 Hotel Booking Customer Segmentation")


lead_time = st.number_input("⏳ Lead Time (days before arrival)", min_value=0)
adr = st.number_input("💰 Average Daily Rate (ADR)", min_value=0.0)
week_nights = st.number_input("🛏️ Stays in Week Nights", min_value=0)
weekend_nights = st.number_input("🌙 Stays in Weekend Nights", min_value=0)
adults = st.number_input("👨‍👩‍👧 Number of Adults", min_value=1)
children = st.number_input("🧒 Number of Children", min_value=0)
babies = st.number_input("👶 Number of Babies", min_value=0)


if st.button("🔍 Predict Customer Segment"):
    if adults + children + babies == 0:
        st.warning("⚠️ At least one guest (adult, child, or baby) must be present.")
    else:
        # Prepare data
        input_data = pd.DataFrame([[lead_time, adr, week_nights, weekend_nights, adults, children, babies]],
                                  columns=['lead_time', 'adr', 'stays_in_week_nights',
                                           'stays_in_weekend_nights', 'adults', 'children', 'babies'])
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        cluster = model.predict(scaled_input)[0]
        
        # Get label
        segment = segment_labels.get(cluster, f"Segment {cluster}")
        
        # Display result
        st.success(f"✅ This booking belongs to **{segment}** (Cluster {cluster})")






