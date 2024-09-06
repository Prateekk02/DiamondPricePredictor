import streamlit as st
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

# Create a Streamlit app
st.title("Diamond Price Prediction App")

# Create a form to collect user input
with st.form("diamond_features"):
    carat = st.number_input("Carat:")
    depth = st.number_input("Depth:")
    table = st.number_input("Table:")
    x = st.number_input("X:")
    y = st.number_input("Y:")
    z = st.number_input("Z:")
    cut = st.selectbox("Cut:", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color:", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity:", ["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "IF"])

    # Create a CustomData object from user input
    custom_data = CustomData(carat, depth, table, x, y, z, cut, color, clarity)

    # Create a PredictPipeline object
    pipeline = PredictPipeline()

    # Make a prediction when the form is submitted
    if st.form_submit_button("Make Prediction"):
        # Get the input data as a Pandas DataFrame
        input_df = custom_data.get_data_as_dataframe()

        # Make a prediction using the PredictPipeline
        prediction = pipeline.predict(input_df)

        # Display the prediction
        st.write("Predicted Price:", prediction[0])