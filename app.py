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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
from lime import lime_tabular

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

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
    except:
        return raw_output, visuals

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

def format_model_evaluation(metrics_dict, feature_importance_path=None, explainability_path=None):
    report = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
        <h1 style="color: #2B547E; border-bottom: 2px solid #2B547E; padding-bottom: 10px;">🧠 Model Evaluation Report</h1>
        
        <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #2B547E;">📈 Performance Metrics</h2>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin: 0 0 10px 0; color: #4A708B;">Accuracy</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics_dict.get('accuracy', 'N/A'):.4f}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin: 0 0 10px 0; color: #4A708B;">Precision</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics_dict.get('precision', 'N/A'):.4f}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin: 0 0 10px 0; color: #4A708B;">Recall</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics_dict.get('recall', 'N/A'):.4f}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin: 0 0 10px 0; color: #4A708B;">F1 Score</h3>
                    <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics_dict.get('f1', 'N/A'):.4f}</p>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h2 style="color: #2B547E;">📊 Feature Importance & Explainability</h2>
            {f'<img src="/file={feature_importance_path}" style="max-width: 100%; height: auto; margin-top: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">' if feature_importance_path else ''}
            {f'<img src="/file={explainability_path}" style="max-width: 100%; height: auto; margin-top: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">' if explainability_path else ''}
        </div>
        
        <div style="margin-top: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #2B547E;">🔄 Hyperparameters</h2>
            <pre style="margin: 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">{metrics_dict.get('best_params', 'N/A')}</pre>
        </div>
    </div>
    """
    return report

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

def preprocess_features(data, target_column, feature_engineering=True):
    """
    Preprocess features with optional feature engineering
    """
    # Check if data is loaded
    if data is None or not isinstance(data, pd.DataFrame):
        return None, None, None, None, None
    
    # Separate features and target
    if target_column not in data.columns:
        # Try to infer target column if it's not specified
        for col in ['target', 'label', 'class', 'outcome', 'y']:
            if col in data.columns:
                target_column = col
                break
        else:
            return None, None, None, None, None
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Basic preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Feature engineering when enabled
    if feature_engineering:
        # Create interaction terms between numerical features
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                if len(numerical_cols) > 1:
                    X_train[f'{col1}_{col2}_interaction'] = X_train[col1] * X_train[col2]
                    X_test[f'{col1}_{col2}_interaction'] = X_test[col1] * X_test[col2]
        
        # Create polynomial features for numerical columns (quadratic terms)
        for col in numerical_cols:
            X_train[f'{col}_squared'] = X_train[col] ** 2
            X_test[f'{col}_squared'] = X_test[col] ** 2
        
        # Create aggregate features for categorical columns
        for col in categorical_cols:
            # For each categorical column, calculate mean of numerical columns grouped by categories
            for num_col in numerical_cols:
                if num_col in X_train.columns:
                    agg_map = X_train.groupby(col)[num_col].mean().to_dict()
                    X_train[f'{col}_{num_col}_agg'] = X_train[col].map(agg_map)
                    X_test[f'{col}_{num_col}_agg'] = X_test[col].map(agg_map)
    
    return X_train, X_test, y_train, y_test, preprocessor

def create_shap_plot(model, X_test, feature_names):
    """Create SHAP summary plot for model explainability"""
    plt.figure(figsize=(12, 8))
    
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use the positive class
            
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    else:
        # Fallback for non-tree models
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 50))
        shap_values = explainer.shap_values(X_test[:50])
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use the positive class
            
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
    
    plt.tight_layout()
    file_path = './figures/shap_summary.png'
    plt.savefig(file_path)
    plt.close()
    return file_path

def create_lime_explanation(model, X_train, X_test, feature_names):
    """Create LIME explanation for a sample instance"""
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Negative", "Positive"],
        mode="classification"
    )
    
    # Explain a sample instance
    instance_idx = 0
    exp = explainer.explain_instance(
        X_test[instance_idx], 
        model.predict_proba,
        num_features=10
    )
    
    # Plot explanation
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.tight_layout()
    file_path = './figures/lime_explanation.png'
    plt.savefig(file_path)
    plt.close()
    return file_path

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot if model supports it"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        file_path = './figures/feature_importance.png'
        plt.savefig(file_path)
        plt.close()
        return file_path
    return None

def train_and_evaluate_model(csv_file, target_column, model_type, feature_eng_enabled=True, explainer_type="shap"):
    """Train, evaluate model with metrics and explainability"""
    if not csv_file:
        return "Please upload a CSV file", None, []
    
    # Load data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        return f"Error loading data: {str(e)}", None, []
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_features(
        data, target_column, feature_engineering=feature_eng_enabled
    )
    
    if X_train is None:
        return f"Error: Could not identify target column '{target_column}'", None, []
    
    # Apply preprocessing
    X_train_processed = X_train
    X_test_processed = X_test
    
    # Select model
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:  # Default to gradient boosting
        model = GradientBoostingClassifier(random_state=42)
    
    # Train model
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
    }
    
    # Generate feature names
    feature_names = X_train_processed.columns.tolist()
    
    # Create feature importance plot
    feature_importance_path = create_feature_importance_plot(model, feature_names)
    
    # Create explainability visualization
    explainability_path = None
    if explainer_type == "shap":
        explainability_path = create_shap_plot(model, X_test_processed, feature_names)
    else:  # LIME
        explainability_path = create_lime_explanation(model, X_train_processed.values, 
                                                     X_test_processed.values, feature_names)
    
    # Log to wandb
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="huggingface-model-evaluation", config={
        "model_type": model_type,
        "feature_engineering": feature_eng_enabled,
        "explainer": explainer_type,
        "metrics": metrics
    })
    
    wandb.log(metrics)
    
    if feature_importance_path:
        wandb.log({"feature_importance": wandb.Image(feature_importance_path)})
    
    if explainability_path:
        wandb.log({"explainability": wandb.Image(explainability_path)})
    
    run.finish()
    
    # Return results
    results = [feature_importance_path, explainability_path] if feature_importance_path and explainability_path else []
    return format_model_evaluation(metrics, feature_importance_path, explainability_path), None, results

def objective(trial, csv_file, target_column, model_type, feature_eng_enabled=True):
    """Objective function for Optuna hyperparameter optimization"""
    try:
        # Load data
        data = pd.read_csv(csv_file)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_features(
            data, target_column, feature_engineering=feature_eng_enabled
        )
        
        if X_train is None:
            return 0.0
        
        # Apply preprocessing
        X_train_processed = X_train
        X_test_processed = X_test
        
        # Hyperparameters based on model type
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                random_state=42
            )
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=42
            )
        
        # Train model
        model.fit(X_train_processed, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_processed)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return f1
    
    except Exception as e:
        print(f"Error in objective function: {str(e)}")
        return 0.0

def tune_hyperparameters(csv_file, target_column, model_type, n_trials=10, feature_eng_enabled=True):
    """Run hyperparameter tuning with Optuna"""
    if not csv_file:
        return "Please upload a CSV file first"
    
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    run = wandb.init(project="huggingface-hyperparameter-tuning", config={
        "model_type": model_type,
        "feature_engineering": feature_eng_enabled,
        "n_trials": n_trials
    })
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, csv_file, target_column, model_type, feature_eng_enabled), 
        n_trials=n_trials
    )
    
    # Log best parameters to wandb
    wandb.log({"best_params": study.best_params, "best_value": study.best_value})
    
    # Visualization of optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    history_path = './figures/optuna_history.png'
    plt.savefig(history_path)
    plt.close()
    
    # Visualization of parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    importance_path = './figures/optuna_importance.png'
    plt.savefig(importance_path)
    plt.close()
    
    # Log visualizations
    wandb.log({"optimization_history": wandb.Image(history_path)})
    wandb.log({"parameter_importance": wandb.Image(importance_path)})
    
    run.finish()
    
    # Return a formatted result
    result = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
        <h1 style="color: #2B547E; border-bottom: 2px solid #2B547E; padding-bottom: 10px;">⚙️ Hyperparameter Optimization Results</h1>
        
        <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <h2 style="color: #2B547E;">🏆 Best Parameters</h2>
            <pre style="margin: 10px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">{study.best_params}</pre>
            
            <h3 style="color: #4A708B; margin-top: 20px;">Best F1 Score</h3>
            <p style="font-size: 20px; font-weight: bold;">{study.best_value:.4f}</p>
        </div>
        
        <div style="margin-top: 30px;">
            <h2 style="color: #2B547E;">📈 Optimization Results</h2>
            <img src="/file={history_path}" style="max-width: 100%; height: auto; margin-top: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <img src="/file={importance_path}" style="max-width: 100%; height: auto; margin-top: 10px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        </div>
    </div>
    """
    
    # Return results and visualization paths for gallery
    return result, [history_path, importance_path]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📊 AI Data Analysis & ML Experimentation Platform")
    
    with gr.Tab("Data Analysis"):
        with gr.Row():
            with gr.Column():
                file_input_analysis = gr.File(label="Upload CSV Dataset", type="filepath")
                notes_input = gr.Textbox(label="Dataset Notes (Optional)", lines=3)
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                analysis_output = gr.HTML("### Analysis results will appear here...")
                gallery = gr.Gallery(label="Data Visualizations", columns=2)
        
        analyze_btn.click(fn=analyze_data, inputs=[file_input_analysis, notes_input], outputs=[analysis_output, gallery])
    
    with gr.Tab("ML Model Experimentation"):
        with gr.Row():
            with gr.Column():
                file_input_model = gr.File(label="Upload CSV Dataset", type="filepath")
                target_column = gr.Textbox(label="Target Column Name", placeholder="e.g., target, class, outcome")
                model_type = gr.Radio(["random_forest", "gradient_boosting"], label="Model Type", value="random_forest")
                feature_eng = gr.Checkbox(label="Enable Feature Engineering", value=True)
                explainer_type = gr.Radio(["shap", "lime"], label="Explainability Tool", value="shap")
                train_btn = gr.Button("Train & Evaluate Model", variant="primary")
            with gr.Column():
                model_output = gr.HTML("### Model evaluation results will appear here...")
                model_metrics = gr.Textbox(label="Raw Metrics", visible=False)
                model_gallery = gr.Gallery(label="Model Visualizations", columns=2)
        
        train_btn.click(
            fn=train_and_evaluate_model, 
            inputs=[file_input_model, target_column, model_type, feature_eng, explainer_type], 
            outputs=[model_output, model_metrics, model_gallery]
        )
    
    with gr.Tab("Hyperparameter Tuning"):
        with gr.Row():
            with gr.Column():
                file_input_hp = gr.File(label="Upload CSV Dataset", type="filepath")
                target_column_hp = gr.Textbox(label="Target Column Name", placeholder="e.g., target, class, outcome")
                model_type_hp = gr.Radio(["random_forest", "gradient_boosting"], label="Model Type", value="random_forest")
                feature_eng_hp = gr.Checkbox(label="Enable Feature Engineering", value=True)
                n_trials = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Optimization Trials")
                tune_btn = gr.Button("Run Hyperparameter Optimization", variant="primary")
            with gr.Column():
                hp_output = gr.HTML("### Hyperparameter tuning results will appear here...")
                hp_gallery = gr.Gallery(label="Optimization Visualizations", columns=2)
    
        tune_btn.click(
            fn=tune_hyperparameters, 
            inputs=[file_input_hp, target_column_hp, model_type_hp, n_trials, feature_eng_hp], 
            outputs=[hp_output, hp_gallery]
        )

demo.launch(debug=True)