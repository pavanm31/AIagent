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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report)
from lime.lime_text import LimeTextExplainer
from functools import lru_cache
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

def detect_target_column(df):
    """Try to automatically detect the target column"""
    # Common target column names
    possible_targets = ['target', 'label', 'class', 'y', 'outcome', 'result']
    
    for col in possible_targets:
        if col in df.columns:
            return col
    
    # If none found, return the last column by default
    return df.columns[-1]

@lru_cache(maxsize=10)
def cached_generate_lime_explanation(insight_text: str, class_names: tuple = ("Negative", "Positive")):
    """Generate and cache LIME explanations to improve performance"""
    explainer = LimeTextExplainer(class_names=class_names)
    
    def classifier_fn(texts):
        responses = []
        for text in texts:
            prompt = f"""
            Analyze the following data insight and classify its sentiment:
            Insight: {text}
            
            Return response as a JSON format with 'positive' and 'negative' scores:
            {{"positive": 0.0-1.0, "negative": 0.0-1.0}}
            """
            response = model.generate(prompt, max_tokens=100)
            try:
                response_dict = ast.literal_eval(response)
                pos = float(response_dict.get("positive", 0))
                neg = float(response_dict.get("negative", 0))
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

def generate_shap_explanation(model, X_train, X_test):
    """Generate SHAP explanations for model predictions"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Save SHAP plots
        shap_figures = []
        for plot_type in ['summary', 'bar']:
            plt.figure()
            if plot_type == 'summary':
                shap.summary_plot(shap_values, X_test, plot_size=(10, 8), show=False)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            
            fig_path = f'./figures/shap_{plot_type}.png'
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
            shap_figures.append(fig_path)
        
        return shap_figures
    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
        return []

def feature_engineering_experiments(X_train, X_test, y_train, y_test):
    """Run different feature engineering approaches and compare results"""
    results = {}
    
    try:
        # Original features baseline
        base_model = RandomForestClassifier(random_state=42)
        base_model.fit(X_train, y_train)
        y_pred = base_model.predict(X_test)
        results['baseline'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Standardized features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scaled_model = RandomForestClassifier(random_state=42)
        scaled_model.fit(X_train_scaled, y_train)
        y_pred = scaled_model.predict(X_test_scaled)
        results['scaled'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Polynomial features (only if few features)
        if X_train.shape[1] < 10:
            poly = PolynomialFeatures(degree=2, interaction_only=True)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            
            poly_model = RandomForestClassifier(random_state=42)
            poly_model.fit(X_train_poly, y_train)
            y_pred = poly_model.predict(X_test_poly)
            results['polynomial'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        
        # Feature selection
        k = min(5, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_model = RandomForestClassifier(random_state=42)
        selected_model.fit(X_train_selected, y_train)
        y_pred = selected_model.predict(X_test_selected)
        results['selected'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
    except Exception as e:
        print(f"Error in feature engineering experiments: {e}")
    
    return results

def analyze_data(csv_file, additional_notes=""):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2
    
    if os.path.exists('./figures'):
        shutil.rmtree('./figures')
    os.makedirs('./figures', exist_ok=True)
    
    try:
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        run = wandb.init(project="huggingface-data-analysis", config={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "additional_notes": additional_notes,
            "source_file": csv_file.name if csv_file else None
        })
    except:
        run = None
    
    try:
        # Load and preprocess data
        data = pd.read_csv(csv_file.name)
        target_col = detect_target_column(data)
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature engineering experiments
        feat_eng_results = feature_engineering_experiments(X_train, X_test, y_train, y_test)
        if run:
            wandb.log({"feature_engineering": feat_eng_results})
        
        # Train final model with best approach (using baseline here for demo)
        final_model = RandomForestClassifier(random_state=42)
        final_model.fit(X_train, y_train)
        
        # Generate SHAP explanations
        shap_figs = generate_shap_explanation(final_model, X_train, X_test)
        
        agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"])
        analysis_result = agent.run(f"""
            You are an expert data analyst. Perform comprehensive analysis including:
            1. Basic statistics and data quality checks
            2. Feature engineering experiment results: {feat_eng_results}
            3. Target column used: {target_col}
            4. 3 insightful analytical questions about relationships in the data
            5. Visualization of key patterns and correlations
            6. Actionable real-world insights derived from findings
            Generate publication-quality visualizations and save to './figures/'
        """, additional_args={"additional_notes": additional_notes, "source_file": csv_file})
        
        execution_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 ** 2
        memory_usage = final_memory - initial_memory
        if run:
            wandb.log({"execution_time_sec": execution_time, "memory_usage_mb": memory_usage})
        
        visuals = [os.path.join('./figures', f) for f in os.listdir('./figures') if f.endswith(('.png', '.jpg', '.jpeg'))]
        visuals.extend(shap_figs)  # Add SHAP visualizations
        
        if run:
            for viz in visuals:
                wandb.log({os.path.basename(viz): wandb.Image(viz)})
        
        if run:
            run.finish()
        return format_analysis_report(analysis_result, visuals)
    
    except Exception as e:
        if run:
            run.finish()
        return f"Error analyzing data: {str(e)}", [], []

def tune_hyperparameters(csv_file, n_trials: int):
    """Run hyperparameter optimization with Optuna"""
    try:
        data = pd.read_csv(csv_file.name)
        target_col = detect_target_column(data)
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
        
        # Train final model with best params
        best_model = RandomForestClassifier(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return f"Best Hyperparameters: {study.best_params}\n\nValidation Metrics:\n{metrics}"
    except Exception as e:
        return f"Error tuning hyperparameters: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis Agent with Explainability")
    
    insights_store = gr.State([])
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV Dataset", type="filepath")
            notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
            analyze_btn = gr.Button("Analyze", variant="primary")
            optuna_trials = gr.Number(label="Number of Hyperparameter Tuning Trials", value=10)
            tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
            
            insight_dropdown = gr.Dropdown(
                label="Select Insight to Explain",
                interactive=True,
                visible=False
            )
            explain_btn = gr.Button("Generate Explanation", variant="primary", visible=False)
            
        with gr.Column():
            analysis_output = gr.HTML("### Analysis results will appear here...")
            optuna_output = gr.Textbox(label="Optimization Results")
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
    
    analyze_btn.click(
        fn=analyze_data,
        inputs=[file_input, notes_input],
        outputs=[analysis_output, gallery, insights_store]
    ).then(
        fn=update_insight_dropdown,
        inputs=insights_store,
        outputs=[insight_dropdown, explain_btn]
    )
    
    explain_btn.click(
        fn=generate_explanation,
        inputs=insight_dropdown,
        outputs=explanation_html
    )
    
    tune_btn.click(
        fn=tune_hyperparameters,
        inputs=[file_input, optuna_trials],
        outputs=[optuna_output]
    )

demo.launch(debug=True, share=True)