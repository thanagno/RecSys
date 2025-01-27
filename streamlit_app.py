import streamlit as st
import pandas as pd
import requests

# Streamlit UI
st.title("Movie Recommender")
st.write("Provide user details to get personalized movie recommendations.")

# Input form
with st.form("recommendation_form"):
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    neighbors_num = st.number_input("Number of Neighbors", min_value=1, value=10)
    recommendations_num = st.number_input("Number of Recommendations", min_value=1, value=5)
    submit = st.form_submit_button("Get Recommendations")

# Process input
if submit:
    # API endpoint
    api_url = "http://localhost:8000/api/v1/recommendations/"  # Update with your FastAPI server URL
    
    # Payload for FastAPI
    payload = {
        "user_id": user_id,
        "neighbors_num": neighbors_num,
        "recommendations_num": recommendations_num
    }
    
    try:
        # Send request to FastAPI
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        recommendations = response.json()

        st.success(f"Top Recommendations for User {user_id}:")
        df = pd.DataFrame(recommendations)
        st.write(df)
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {str(e)}")
