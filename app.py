import gradio as gr
import pandas as pd
import smolagents
import shap
import optuna
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def preprocess_data(file):
    df = pd.read_csv(file.name) if file.name.endswith(".csv") else pd.read_excel(file.name)
    df = smolagents.clean(df)  # SmolAgent for preprocessing (handling missing values, fixing format errors, duplicates, etc.)
    return df

def generate_insights(df):
    insights = smolagents.analyze(df)  # SmolAgent to generate data insights
    return insights

def generate_visualizations(df):
    figs = smolagents.visualize(df)  # SmolAgent to create key visualizations
    return figs

def objective(trial, X_train, y_train, X_test, y_test):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_model(df):
    X = df.drop(columns=[df.columns[-1]])
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)
    best_params = study.best_params
    
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)
    return best_model, X_test, y_test, best_params

def explain_model(model, X_test):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap_fig = shap.summary_plot(shap_values, X_test, show=False)
    return shap_fig

def track_experiment(best_params):
    wandb.init(project='data_analysis_ai_agent', reinit=True)
    wandb.config.update(best_params)
    wandb.log({"message": "Tracking Experiment with Optuna tuning"})
    return "Experiment Tracked with Weights & Biases"

def main(file):
    df = preprocess_data(file)
    insights = generate_insights(df)
    figs = generate_visualizations(df)
    model, X_test, y_test, best_params = train_model(df)
    shap_fig = explain_model(model, X_test)
    tracking_msg = track_experiment(best_params)
    
    return insights, figs, shap_fig, tracking_msg

iface = gr.Interface(fn=main, inputs=gr.File(), outputs=["text", gr.Plot(), gr.Plot(), "text"])
iface.launch()
