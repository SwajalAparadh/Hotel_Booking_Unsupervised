
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Hotel Booking Customer Segmentation")

lead_time = st.number_input("Lead Time", min_value=0)
adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0)
week_nights = st.number_input("Stays in Week Nights", min_value=0)
weekend_nights = st.number_input("Stays in Weekend Nights", min_value=0)
adults = st.number_input("Number of Adults", min_value=1)
children = st.number_input("Number of Children", min_value=0)
babies = st.number_input("Number of Babies", min_value=0)

if st.button("Predict Customer Segment"):
    input_data = pd.DataFrame([[lead_time, adr, week_nights, weekend_nights, adults, children, babies]],
                              columns=['lead_time', 'adr', 'stays_in_week_nights',
                                       'stays_in_weekend_nights', 'adults', 'children', 'babies'])
    
    scaled_input = scaler.transform(input_data)
    cluster = model.predict(scaled_input)[0]
    
    st.success(f"This booking belongs to Customer Segment {cluster}")
