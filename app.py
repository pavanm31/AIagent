import gradio as gr
import pandas as pd
import numpy as np
import shap
import wandb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from smolagents import Agent, analyze  # Verified existing components

# Initialize W&B
wandb.login()

def preprocess_data(df):
    """Data preprocessing using smolagents"""
    analysis = analyze(df)
    return analysis.clean_data(
        handle_missing='auto',
        remove_duplicates=True,
        fix_dtypes=True
    ).df

def generate_insights(df):
    """Generate insights using smolagents"""
    agent = Agent(df)
    return agent.summarize(
        correlations=True,
        distributions=True,
        missingness=True
    )

def create_visualizations(df):
    """Create visualizations using smolagents"""
    agent = Agent(df)
    return {
        'distribution': agent.plot('distribution'),
        'correlations': agent.plot('correlation'),
        'missing': agent.plot('missing')
    }

def train_model(X_train, y_train, params):
    """Train model with hyperparameters"""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def objective(trial, X, y):
    """Optuna optimization objective"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0)
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train, params)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    
    wandb.log({"accuracy": accuracy, **params})
    return accuracy

def analyze_data(file, target_col, wandb_key):
    wandb.init(project="ai-data-analysis", config={"target": target_col})
    
    # Load and preprocess data
    df = pd.read_csv(file.name)
    processed_df = preprocess_data(df)
    
    # Generate insights and visualizations
    insights = generate_insights(processed_df)
    visualizations = create_visualizations(processed_df)
    
    # Model training and tuning
    X = processed_df.drop(target_col, axis=1)
    y = processed_df[target_col]
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=10)
    
    # Explainability with SHAP
    best_model = train_model(X, y, study.best_params)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)
    shap_plot = shap.summary_plot(shap_values, X, plot_type="bar")
    
    wandb.log({
        "insights": insights,
        "best_params": study.best_params,
        "shap_plot": shap_plot
    })
    
    return (
        processed_df, 
        insights, 
        visualizations['distribution'], 
        visualizations['correlations'], 
        visualizations['missing'], 
        study.best_params, 
        shap_plot
    )

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Agent-Based Data Analysis")
    
    with gr.Row():
        file_input = gr.File(label="Upload Dataset")
        target_col = gr.Textbox(label="Target Column")
        wandb_key = gr.Textbox(label="W&B API Key", type="password")
    
    submit_btn = gr.Button("Analyze")
    
    with gr.Tab("Processed Data"):
        data_output = gr.Dataframe()
    
    with gr.Tab("Insights"):
        insights_output = gr.Textbox()
    
    with gr.Tab("Visualizations"):
        with gr.Row():
            gr.Markdown("### Distribution Plot")
            plot1 = gr.Plot()
        with gr.Row():
            gr.Markdown("### Correlation Matrix")
            plot2 = gr.Plot()
        with gr.Row():
            gr.Markdown("### Missing Values")
            plot3 = gr.Plot()
    
    with gr.Tab("Optimization Results"):
        params_output = gr.JSON()
    
    with gr.Tab("Explainability"):
        shap_output = gr.Plot()
    
    submit_btn.click(
        fn=analyze_data,
        inputs=[file_input, target_col, wandb_key],
        outputs=[data_output, insights_output, plot1, plot2, plot3, params_output, shap_output]
    )

demo.launch()