import gradio as gr
from smolagents import HfApiModel, CodeAgent
from huggingface_hub import login
import os
import shutil
import wandb
import time
import psutil
import optuna
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

def format_analysis_report(raw_output, visuals):
    try:
        if isinstance(raw_output, dict):
            analysis_dict = raw_output
        else:
            try:
                analysis_dict = ast.literal_eval(str(raw_output))
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing CodeAgent output: {e}")
                return str(raw_output), visuals  # Return raw output as string
                
        report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h1 style="color: #2B547E; border-bottom: 2px solid #2B547E; padding-bottom: 10px;">📊 Data Analysis Report</h1>
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">🔍 Key Observations</h2>
                {format_observations(analysis_dict.get('observations', {}))}
            </div>
            <div style="margin-top: 30px;">
                <h2 style="color: #2B547E;">💡 Insights & Visualizations</h2>
                {format_insights(analysis_dict.get('insights', {}), visuals)}
            </div>
        </div>
        """
        return report, visuals
    except Exception as e:
        print(f"Error in format_analysis_report: {e}")
        return str(raw_output), visuals

def format_observations(observations):
    return '\n'.join([
        f"""
        <div style="margin: 15px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h3 style="margin: 0 0 10px 0; color: #4A708B;">{key.replace('_', ' ').title()}</h3>
            <pre style="margin: 0; padding: 10px; background: #f8f9fa; border-radius: 4px;">{value}</pre>
        </div>
        """ for key, value in observations.items() if 'proportions' in key
    ])

def format_insights(insights, visuals):
    return '\n'.join([
        f"""
        <div style="margin: 20px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #2B547E; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">{idx+1}</div>
                <p style="margin: 0; font-size: 16px;">{insight}</p>
            </div>
            {f'<img src="/file={visuals[idx]}" style="max-width: 100%; height: auto; margin-top: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">' if idx < len(visuals) else ''}
        </div>
        """ for idx, (key, insight) in enumerate(insights.items())
    ])

def analyze_data(csv_file, additional_notes=""):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2
    
    if os.path.exists('./figures'):
        shutil.rmtree('./figures')
    os.makedirs('./figures', exist_ok=True)
    
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="huggingface-data-analysis", config={
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "additional_notes": additional_notes,
        "source_file": csv_file.name if csv_file else None
    })
    
    agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn", "sklearn"])
    analysis_result = agent.run("""
        You are an expert data analyst. Perform comprehensive analysis including:
        1. Basic statistics and data quality checks
        2. 3 insightful analytical questions about relationships in the data
        3. Visualization of key patterns and correlations
        4. Actionable real-world insights derived from findings.
        Generate publication-quality visualizations and save to './figures/'.
        Return the analysis results as a python dictionary that can be parsed by ast.literal_eval().
        The dictionary should have the following structure:
        {
            'observations': {
                'observation_1_key': 'observation_1_value',
                'observation_2_key': 'observation_2_value',
                ...
            },
            'insights': {
                'insight_1_key': 'insight_1_value',
                'insight_2_key': 'insight_2_value',
                ...
            }
        }
    """, additional_args={"additional_notes": additional_notes, "source_file": csv_file})
    
    execution_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 ** 2
    memory_usage = final_memory - initial_memory
    wandb.log({"execution_time_sec": execution_time, "memory_usage_mb": memory_usage})
    
    visuals = [os.path.join('./figures', f) for f in os.listdir('./figures') if f.endswith(('.png', '.jpg', '.jpeg'))]
    for viz in visuals:
        wandb.log({os.path.basename(viz): wandb.Image(viz)})
    
    run.finish()
    return format_analysis_report(analysis_result, visuals)

def objective(trial, X_train, y_train, X_test, y_test):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def tune_hyperparameters(csv_file, n_trials: int):
    df = pd.read_csv(csv_file)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="maximize")
    objective_func = lambda trial: objective(trial, X_train, y_train, X_test, y_test)
    study.optimize(objective_func, n_trials=n_trials)
    
    best_params = study.best_params
    best_value = study.best_value
    
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    wandb.log({
        "best_params": best_params,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_fig_path = "./figures/shap_summary.png"
    plt.savefig(shap_fig_path)
    wandb.log({"shap_summary": wandb.Image(shap_fig_path)})
    plt.clf()
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['target'], mode='classification')
    lime_explanation = lime_explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
    lime_fig = lime_explanation.as_pyplot_figure()
    lime_fig_path = "./figures/lime_explanation.png"
    lime_fig.savefig(lime_fig_path)
    wandb.log({"lime_explanation": wandb.Image(lime_fig_path)})
    plt.clf()

    return f"Best Hyperparameters: {best_params}<br>Accuracy: {accuracy}<br>Precision: {precision}<br>Recall: {recall}<br>F1-score: {f1}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis Agent with Hyperparameter Optimization")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV Dataset", type="filepath")
            notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
            analyze_btn = gr.Button("Analyze", variant="primary")
            optuna_trials = gr.Number(label="Number of Hyperparameter Tuning Trials", value=10)
            tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
        with gr.Column():
            analysis_output = gr.Markdown("### Analysis results will appear here...")
            optuna_output = gr.HTML(label="Hyperparameter Tuning Results")
            gallery = gr.Gallery(label="Data Visualizations", columns=2)
    
    analyze_btn.click(fn=analyze_data, inputs=[file_input, notes_input], outputs=[analysis_output, gallery])
    tune_btn.click(fn=tune_hyperparameters, inputs=[file_input, optuna_trials], outputs=[optuna_output])

demo.launch(debug=True)