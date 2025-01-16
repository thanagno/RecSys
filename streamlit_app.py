import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('data/models/user neighbors.pkl')  # Update the path as necessary

# Streamlit UI
st.title("Movie Recommender")
st.write("Provide user details to get personalized movie recommendations.")

# Input form
with st.form("recommendation_form"):
    user_id = st.number_input("Enter User ID", min_value=1, step=1)
    submit = st.form_submit_button("Get Recommendations")

# Process input
if submit:
    # Assuming the model has a function `get_recommendations`
    try:
        recommendations = model.get_recommendations(user_id)
        st.success(f"Top Recommendations for User {user_id}:")
        st.write(pd.DataFrame(recommendations, columns=["Movie ID", "Title", "Rating"]))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
