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
from smolagents import tool, CodeAgent, HfApiModel

@tool
def preprocess_data(file):
    df = pd.read_csv(file.name) if file.name.endswith(".csv") else pd.read_excel(file.name)
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

@tool
def generate_insights(df):
    insights = df.describe().to_string()
    return insights

@tool
def generate_visualizations(df):
    figs = []
    for column in df.select_dtypes(include='number').columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        figs.append(fig)
    return figs

def objective(trial, X_train, y_train, X_test, y_test):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

@tool
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

@tool
def explain_model(model, X_test):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap_fig = shap.summary_plot(shap_values, X_test, show=False)
    return shap_fig

@tool
def track_experiment(best_params):
    wandb.init(project='data_analysis_ai_agent', reinit=True)
    wandb.config.update(best_params)
    wandb.log({"message": "Tracking Experiment with Optuna tuning"})
    return "Experiment Tracked with Weights & Biases"

# Initialize CodeAgent with tools
model = HfApiModel(repo_id="bigcode/starcoder")
tools = [preprocess_data, generate_insights, generate_visualizations, train_model, explain_model, track_experiment]
agent = CodeAgent(model=model, tools=tools)

def main(file):
    df = agent.run(f"preprocess_data(file)")
    insights = agent.run(f"generate_insights(df)")
    figs = agent.run(f"generate_visualizations(df)")
    model, X_test, y_test, best_params = agent.run(f"train_model(df)")
    shap_fig = agent.run(f"explain_model(model, X_test)")
    tracking_msg = agent.run(f"track_experiment(best_params)")
    
    return insights, figs, shap_fig, tracking_msg

iface = gr.Interface(fn=main, inputs=gr.File(), outputs=["text", gr.Plot(), gr.Plot(), "text"])
iface.launch()
