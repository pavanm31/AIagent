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
import numpy as np
from lime.lime_text import LimeTextExplainer
from functools import lru_cache

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

# Cache for explanations (last 10 explanations)
@lru_cache(maxsize=10)
def cached_generate_lime_explanation(insight_text: str, class_names: tuple = ("Negative", "Positive")):
    """
    Generate and cache LIME explanations to improve performance
    """
    explainer = LimeTextExplainer(class_names=class_names)
    
    def classifier_fn(texts):
        # Use the actual model to get predictions for explanations
        responses = []
        for text in texts:
            # Create a prompt that asks for sentiment analysis
            prompt = f"""
            Analyze the following data insight and classify its sentiment:
            Insight: {text}
            
            Return response as a JSON format with 'positive' and 'negative' scores:
            {{"positive": 0.0-1.0, "negative": 0.0-1.0}}
            """
            response = model.generate(prompt, max_tokens=100)
            try:
                # Parse the JSON response
                response_dict = ast.literal_eval(response)
                pos = float(response_dict.get("positive", 0))
                neg = float(response_dict.get("negative", 0))
                # Normalize to sum to 1
                total = pos + neg
                if total > 0:
                    pos /= total
                    neg /= total
                responses.append([neg, pos])
            except:
                responses.append([0.5, 0.5])
        return np.array(responses)
    
    exp = explainer.explain_instance(
        insight_text,
        classifier_fn,
        num_features=10,
        top_labels=1,
        num_samples=100
    )
    return exp.as_html()

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
        return report, visuals, list(analysis_dict.get('insights', {}).values()
    except Exception as e:
        print(f"Error formatting report: {e}")
        return raw_output, visuals, []

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

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    return learning_rate * batch_size * num_epochs

def tune_hyperparameters(n_trials: int):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return f"Best Hyperparameters: {study.best_params}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis Agent with Explainability")
    
    # Store insights in a hidden component
    insights_store = gr.State([])
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV Dataset", type="filepath")
            notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
            analyze_btn = gr.Button("Analyze", variant="primary")
            optuna_trials = gr.Number(label="Number of Hyperparameter Tuning Trials", value=10)
            tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
            
            # Add dropdown for insight selection
            insight_dropdown = gr.Dropdown(
                label="Select Insight to Explain",
                interactive=True,
                visible=False
            )
            explain_btn = gr.Button("Generate Explanation", variant="primary", visible=False)
            
        with gr.Column():
            analysis_output = gr.HTML("### Analysis results will appear here...")
            optuna_output = gr.Textbox(label="Best Hyperparameters")
            gallery = gr.Gallery(label="Data Visualizations", columns=2)
            explanation_html = gr.HTML(label="Model Explanation")
    
    def update_insight_dropdown(insights):
        if insights and len(insights) > 0:
            return gr.Dropdown(
                choices=[(f"Insight {i+1}", insight) for i, insight in enumerate(insights)],
                value=insights[0],
                visible=True
            ), gr.Button(visible=True)
        return gr.Dropdown(visible=False), gr.Button(visible=False)
    
    def generate_explanation(selected_insight):
        if not selected_insight:
            return "<p>Please select an insight first</p>"
        return cached_generate_lime_explanation(selected_insight)
    
    # Analysis button click handler
    analyze_btn.click(
        fn=analyze_data,
        inputs=[file_input, notes_input],
        outputs=[analysis_output, gallery, insights_store]
    ).then(
        fn=update_insight_dropdown,
        inputs=insights_store,
        outputs=[insight_dropdown, explain_btn]
    )
    
    # Explanation button click handler
    explain_btn.click(
        fn=generate_explanation,
        inputs=insight_dropdown,
        outputs=explanation_html
    )
    
    # Hyperparameter tuning button
    tune_btn.click(
        fn=tune_hyperparameters,
        inputs=[optuna_trials],
        outputs=[optuna_output]
    )

demo.launch(debug=True)