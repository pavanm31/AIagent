import streamlit as st
import pandas as pd
import numpy as np
from smolagents import DataCleanser, InsightGenerator, DataVisualizer

# Title of the app
st.title("Data Analysis with Hugging Face SmolAgents")

# Step 1: Create a user interface to receive data file
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    st.write("### Raw Data")
    st.write(df)

    # Step 2: Use Hugging Face SmolAgents for data cleansing and preprocessing
    st.write("### Data Cleansing and Preprocessing")
    cleanser = DataCleanser()
    df_cleaned = cleanser.clean_data(df)
    st.write("Cleaned Data:")
    st.write(df_cleaned)

    # Step 3: Use Hugging Face SmolAgents to generate insights
    st.write("### Key Insights from Data")
    insight_generator = InsightGenerator()
    insights = insight_generator.generate_insights(df_cleaned)
    st.write(insights)

    # Step 4: Create data visualizations
    st.write("### Data Visualizations")
    visualizer = DataVisualizer()
    
    # Example visualizations
    st.write("#### Histogram of Numerical Columns")
    numerical_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        fig = visualizer.plot_histogram(df_cleaned, col)
        st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    fig = visualizer.plot_correlation_heatmap(df_cleaned)
    st.pyplot(fig)

    # Step 5: Display the output
    st.write("### Final Output")
    st.write("Data analysis completed. Check the insights and visualizations above.")

else:
    st.write("Please upload a file to get started.")