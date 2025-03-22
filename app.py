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
from functools import lru_cache
import re

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

# Error Handling: Validate CSV file
def validate_csv_file(file_path):
    if not file_path.endswith('.csv'):
        raise ValueError("Invalid file type. Please upload a CSV file.")
    if re.search(r"[^a-zA-Z0-9_\-\.]", file_path):
        raise ValueError("Invalid file name. Please use alphanumeric characters only.")
    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB limit
        raise ValueError("File size exceeds the limit of 100MB.")

# Caching: Cache repeated analyses
@lru_cache(maxsize=5)
def analyze_data_cached(csv_file_path, additional_notes=""):
    return analyze_data(csv_file_path, additional_notes)

# Format Analysis Report
def format_analysis_report(raw_output, visuals):
    try:
        analysis_dict = raw_output if isinstance(raw_output, dict) else ast.literal_eval(str(raw_output))
        
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
        return f"❌ Error formatting report: {str(e)}", visuals

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

# Data Analysis Function
def analyze_data(csv_file, additional_notes=""):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2
    
    try:
        validate_csv_file(csv_file.name)
        
        if os.path.exists('./figures'):
            shutil.rmtree('./figures')
        os.makedirs('./figures', exist_ok=True)
        
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        run = wandb.init(project="huggingface-data-analysis", config={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "additional_notes": additional_notes,
            "source_file": csv_file.name
        })
        
        agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"])
        analysis_result = agent.run("""
            You are an expert data analyst. Perform comprehensive analysis including:
            1. Basic statistics and data quality checks
            2. 3 insightful analytical questions about relationships in the data
            3. Visualization of key patterns and correlations
            4. Actionable real-world insights derived from findings
            Generate publication-quality visualizations and save to './figures/'
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
    
    except Exception as e:
        run.finish() if 'run' in locals() else None
        return f"❌ Error: {str(e)}", []

# Hyperparameter Tuning
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    return learning_rate * batch_size * num_epochs

def tune_hyperparameters(n_trials: int):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Parallel processing
    return f"Best Hyperparameters: {study.best_params}"

# Model Fine-Tuning (Placeholder)
def fine_tune_model(dataset_path, epochs=3, learning_rate=5e-5):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    import pandas as pd

    # Load dataset
    data = pd.read_csv(dataset_path)
    # Preprocess data (placeholder for tokenization and formatting)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    # Fine-tuning logic (simplified)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=learning_rate,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,  # Replace with tokenized dataset
    )
    trainer.train()
    return "Model fine-tuning completed!"

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis Agent with Hyperparameter Optimization")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV Dataset", type="filepath")
            notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
            analyze_btn = gr.Button("Analyze", variant="primary")
            optuna_trials = gr.Number(label="Number of Hyperparameter Tuning Trials", value=10)
            tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
            fine_tune_btn = gr.Button("Fine-Tune Model", variant="secondary")
        with gr.Column():
            analysis_output = gr.Markdown("### Analysis results will appear here...")
            optuna_output = gr.Textbox(label="Best Hyperparameters")
            gallery = gr.Gallery(label="Data Visualizations", columns=2)
    
    analyze_btn.click(fn=analyze_data, inputs=[file_input, notes_input], outputs=[analysis_output, gallery])
    tune_btn.click(fn=tune_hyperparameters, inputs=[optuna_trials], outputs=[optuna_output])
    fine_tune_btn.click(fn=fine_tune_model, inputs=[file_input], outputs=[analysis_output])

demo.launch(debug=True)