# House Price Prediction and Explanation App

This project is a machine learning web application built with **Streamlit** that predicts house prices based on the **Boston House Price Dataset** using an XGBoost regression model. The app also uses OpenAI's language model (GPT-4) to explain the predicted price based on the features provided.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
4. [How to Use](#how-to-use)
5. [Model Performance](#model-performance)
6. [Technologies Used](#technologies-used)
7. [License](#license)

## Project Overview

This web application allows users to input various features of a house and receive a predicted price based on a trained XGBoost model. Additionally, it generates an explanation for the predicted price using OpenAI's language model based on the input features provided by the user. If no valid input is given, the LLM will notify the user that no features were provided.

## Features

- **House Price Prediction**: Predict house prices using a trained machine learning model based on user input features such as median income, house age, number of rooms, and more.
- **Explanation Generation**: The app uses OpenAI's GPT model to explain the predicted price by considering the factors provided.
- **Error Handling**: If no inputs are provided or if they are invalid, the LLM will return an appropriate response without generating explanations.
  
## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/JavedKhaan/Price_Prediction_of_Houses_and_Explanation_App.git
cd Price_Prediction_of_Houses_and_Explanation_App
