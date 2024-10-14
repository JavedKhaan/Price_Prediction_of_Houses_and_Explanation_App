import streamlit as st
import joblib
import numpy as np
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load the trained model
model = joblib.load('trained_model.pkl') # Ensure the path to your saved model is correct

# Streamlit interface
st.title('House Price Prediction and Explanation App')

# Create input fields for each feature
MedInc = st.number_input('Median Income (MedInc)', min_value=0.0, format="%.4f")
HouseAge = st.number_input('House Age (HouseAge)', min_value=0)
AveRooms = st.number_input('Average Number of Rooms (AveRooms)', min_value=0)
AveBedrms = st.number_input('Average Number of Bedrooms (AveBedrms)', min_value=0)
Population = st.number_input('Population', min_value=0)
AveOccup = st.number_input('Average Occupancy (AveOccup)', min_value=0.0, format="%.4f")
Latitude = st.number_input('Latitude', format="%.4f")
Longitude = st.number_input('Longitude', format="%.4f")

# Initialize the LLM with the OpenAI model (e.g., GPT-4)
llm = OpenAI(temperature=0.8, model_name="gpt-4-0-mini")

# When the user presses the 'Predict' button, predict the house price will be display
if st.button('Predict'):# Create a numpy array from the user input
    user_input = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]).reshape(1, -1)
    
    # Predict the price
    prediction = model.predict(user_input)
    # Display the predicted price
    st.success(f"The predicted house price is: ${prediction[0]:,.2f}")

    # Prepare input features and predicted price for the LLM
    input_features = {
        "Median Income": MedInc,
        "House Age": HouseAge,
        "Average Rooms": AveRooms,
        "Average Bedrooms": AveBedrms,
        "Population": Population,
        "Average Occupancy": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
        }

    # Create a prompt for the LLM
    prompt = f"""
    The predicted house price is ${prediction[0]:,.2f}. 
    Based on the following house features: 
    - Median Income: {MedInc}
    - House Age: {HouseAge} years
    - Average Number of Rooms: {AveRooms}
    - Average Number of Bedrooms: {AveBedrms}
    - Population: {Population}
    - Average Occupancy: {AveOccup}
    - Latitude: {Latitude}
    - Longitude: {Longitude}
    Explain the advantages and disadvantages of this house at the predicted price. Include factors related to the features and market context.
    and if all the features are not being filled or zero then do not explain advantages and disadvantages instead respond that there is no any input provided.
    """

    # Get the explanation from the LLM
    llm_response = llm(prompt)  # Use the new `create` method

    # Display the explanation
    st.subheader("Explanation from the LLM")
    st.write(llm_response)