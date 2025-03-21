import gradio as gr
import pandas as pd
import smolagent
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file):
    try:
        df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
        agent = smolagent.SmolAgent()
        cleaned_df = agent.run("Clean and preprocess this dataset", df)
        return cleaned_df.describe().to_string()
    except Exception as e:
        return f"Error in preprocessing: {str(e)}"

def generate_insights(file):
    try:
        df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
        agent = smolagent.SmolAgent()
        insights = agent.run("Generate insights and report on this dataset", df)
        return insights
    except Exception as e:
        return f"Error in generating insights: {str(e)}"

def visualize_data(file):
    try:
        df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
        agent = smolagent.SmolAgent()
        important_features = agent.run("Identify important features in this dataset", df)
        
        if not isinstance(important_features, dict):
            return "Error: Expected a dictionary of feature importance values."
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=list(important_features.keys()), y=list(important_features.values()), ax=ax)
        ax.set_title("Feature Importance")
        plt.xticks(rotation=45)
        return fig
    except Exception as e:
        return f"Error in visualization: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## AI-Powered Data Analysis with SmolAgent")
    file_input = gr.File(label="Upload CSV or Excel File")
    
    preprocess_btn = gr.Button("Preprocess Data")
    preprocess_output = gr.Textbox()
    preprocess_btn.click(preprocess_data, inputs=file_input, outputs=preprocess_output)
    
    insights_btn = gr.Button("Generate Insights")
    insights_output = gr.Textbox()
    insights_btn.click(generate_insights, inputs=file_input, outputs=insights_output)
    
    visualize_btn = gr.Button("Visualize Data")
    visualize_output = gr.Plot()
    visualize_btn.click(visualize_data, inputs=file_input, outputs=visualize_output)
    
demo.launch()
