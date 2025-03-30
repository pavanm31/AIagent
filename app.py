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
import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

def format_analysis_report(raw_output, visuals, metrics=None, explainability_plots=None):
    try:
        analysis_dict = raw_output if isinstance(raw_output, dict) else ast.literal_eval(str(raw_output))
        
        metrics_section = ""
        if metrics:
            metrics_section = f"""
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">📈 Model Performance Metrics</h2>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Accuracy</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics['accuracy']:.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Precision</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics['precision']:.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Recall</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics['recall']:.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">F1 Score</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics['f1']:.2f}</p>
                    </div>
                </div>
            </div>
            """
        
        explainability_section = ""
        if explainability_plots:
            explainability_section = f"""
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">🔍 Model Explainability</h2>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    {''.join([f'<img src="/file={plot}" style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">' for plot in explainability_plots])}
                </div>
            </div>
            """
        
        report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h1 style="color: #2B547E; border-bottom: 2px solid #2B547E; padding-bottom: 10px;">📊 Data Analysis Report</h1>
            {metrics_section}
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">🔍 Key Observations</h2>
                {format_observations(analysis_dict.get('observations', {}))}
            </div>
            <div style="margin-top: 30px;">
                <h2 style="color: #2B547E;">💡 Insights & Visualizations</h2>
                {format_insights(analysis_dict.get('insights', {}), visuals)}
            </div>
            {explainability_section}
        </div>
        """
        return report, visuals
    except:
        return raw_output, visuals

def preprocess_data(df, feature_engineering=True):
    """Handle missing values, categorical encoding, and feature engineering"""
    # Basic preprocessing
    df = df.dropna()
    
    # Convert categorical variables if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:  # One-hot encode if few categories
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df = df.drop(col, axis=1)
    
    # Feature engineering
    if feature_engineering:
        # Create polynomial features for numerical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(df[num_cols])
            poly_cols = [f"poly_{i}" for i in range(poly_features.shape[1])]
            poly_df = pd.DataFrame(poly_features, columns=poly_cols)
            df = pd.concat([df, poly_df], axis=1)
    
    return df

def evaluate_model(X, y, model, test_size=0.2):
    """Evaluate model performance with various metrics"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

def generate_explainability_plots(X, model, feature_names, output_dir='./figures'):
    """Generate SHAP and LIME explainability plots"""
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    # SHAP Analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    plt = shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    shap_path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    plot_paths.append(shap_path)
    
    # LIME Analysis
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X, 
        feature_names=feature_names,
        class_names=['class_0', 'class_1'],  # Update based on your classes
        verbose=True,
        mode='classification'
    )
    
    # Explain a random instance
    exp = explainer.explain_instance(X[0], model.predict_proba, num_features=5)
    lime_path = os.path.join(output_dir, 'lime_explanation.png')
    exp.as_pyplot_figure().savefig(lime_path, bbox_inches='tight')
    plot_paths.append(lime_path)
    
    return plot_paths

def analyze_data(csv_file, additional_notes="", perform_ml=True):
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
        "source_file": csv_file.name if csv_file else None,
        "perform_ml": perform_ml
    })
    
    # Load and preprocess data
    df = pd.read_csv(csv_file)
    processed_df = preprocess_data(df)
    
    metrics = None
    explainability_plots = None
    
    if perform_ml and len(processed_df.columns) > 1:
        try:
            # Assume last column is target for demonstration
            X = processed_df.iloc[:, :-1].values
            y = processed_df.iloc[:, -1].values
            
            # Evaluate baseline model
            baseline_model = RandomForestClassifier(random_state=42)
            metrics = evaluate_model(X, y, baseline_model)
            
            # Generate explainability plots
            feature_names = processed_df.columns[:-1]
            explainability_plots = generate_explainability_plots(X[:100], baseline_model, feature_names)
            
            wandb.log(metrics)
        except Exception as e:
            print(f"ML analysis failed: {str(e)}")
    
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
    return format_analysis_report(analysis_result, visuals, metrics, explainability_plots)

def objective(trial):
    # Define hyperparameter space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    # Load data (you would need to pass this or make it available)
    df = pd.read_csv("temp_data.csv")  # You'll need to handle this properly
    processed_df = preprocess_data(df)
    X = processed_df.iloc[:, :-1].values
    y = processed_df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and evaluate model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Return metric to optimize (F1 score in this case)
    return f1_score(y_test, y_pred, average='weighted')

def tune_hyperparameters(n_trials: int, csv_file):
    try:
        # Save the uploaded file temporarily for Optuna
        if csv_file:
            temp_path = "temp_data.csv"
            with open(temp_path, "wb") as f:
                f.write(csv_file.read())
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            os.remove(temp_path)
            return f"Best Hyperparameters: {study.best_params}\nBest F1 Score: {study.best_value:.4f}"
        else:
            return "Please upload a CSV file first for hyperparameter tuning."
    except Exception as e:
        return f"Hyperparameter tuning failed: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis Agent with Hyperparameter Optimization")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV Dataset", type="filepath")
            notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
            perform_ml = gr.Checkbox(label="Perform Machine Learning Analysis", value=True)
            analyze_btn = gr.Button("Analyze", variant="primary")
            optuna_trials = gr.Number(label="Number of Hyperparameter Tuning Trials", value=10)
            tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
        with gr.Column():
            analysis_output = gr.Markdown("### Analysis results will appear here...")
            optuna_output = gr.Textbox(label="Best Hyperparameters")
            gallery = gr.Gallery(label="Data Visualizations", columns=2)
    
    analyze_btn.click(
        fn=analyze_data,
        inputs=[file_input, notes_input, perform_ml],
        outputs=[analysis_output, gallery]
    )
    tune_btn.click(
        fn=tune_hyperparameters,
        inputs=[optuna_trials, file_input],
        outputs=[optuna_output]
    )

demo.launch(debug=True)