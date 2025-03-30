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
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token, add_to_git_credential=True)

# Initialize Model
model = HfApiModel("mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_token)

def format_observations(observations):
    if not isinstance(observations, dict):
        return f"<pre>{str(observations)}</pre>"
    
    return '\n'.join([
        f"""
        <div style="margin: 15px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h3 style="margin: 0 0 10px 0; color: #4A708B;">{key.replace('_', ' ').title()}</h3>
            <pre style="margin: 0; padding: 10px; background: #f8f9fa; border-radius: 4px;">{value}</pre>
        </div>
        """ for key, value in observations.items()
    ])

def format_insights(insights, visuals):
    if not isinstance(insights, dict):
        return f"<pre>{str(insights)}</pre>"
    
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

def format_analysis_report(raw_output, visuals, metrics=None, explainability_plots=None):
    try:
        # Ensure we have a dictionary to work with
        if isinstance(raw_output, str):
            try:
                analysis_dict = ast.literal_eval(raw_output)
            except:
                analysis_dict = {'observations': {'raw_output': raw_output}, 'insights': {}}
        elif isinstance(raw_output, dict):
            analysis_dict = raw_output
        else:
            analysis_dict = {'observations': {'raw_output': str(raw_output)}, 'insights': {}}

        # Metrics section
        metrics_section = ""
        if metrics:
            metrics_section = f"""
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">📈 Model Performance Metrics</h2>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Accuracy</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics.get('accuracy', 0):.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Precision</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics.get('precision', 0):.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">Recall</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics.get('recall', 0):.2f}</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #4A708B;">F1 Score</h3>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{metrics.get('f1', 0):.2f}</p>
                    </div>
                </div>
            </div>
            """
        
        # Explainability section
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
        
        # Observations section
        observations_section = ""
        if 'observations' in analysis_dict:
            observations_section = f"""
            <div style="margin-top: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #2B547E;">🔍 Key Observations</h2>
                {format_observations(analysis_dict['observations'])}
            </div>
            """
        
        # Insights section
        insights_section = ""
        if 'insights' in analysis_dict:
            insights_section = f"""
            <div style="margin-top: 30px;">
                <h2 style="color: #2B547E;">💡 Insights & Visualizations</h2>
                {format_insights(analysis_dict.get('insights', {}), visuals)}
            </div>
            """
        
        # Build the complete report
        report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h1 style="color: #2B547E; border-bottom: 2px solid #2B547E; padding-bottom: 10px;">📊 Data Analysis Report</h1>
            {metrics_section}
            {explainability_section}
            {observations_section}
            {insights_section}
        </div>
        """
        
        return report, visuals
        
    except Exception as e:
        error_report = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h1 style="color: #B22222;">⚠️ Error Generating Report</h1>
            <p>An error occurred while generating the report:</p>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">{str(e)}</pre>
            <p>Raw output:</p>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">{str(raw_output)}</pre>
        </div>
        """
        return error_report, visuals

def preprocess_data(df, feature_engineering=True):
    """Handle missing values, categorical encoding, and feature engineering"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Basic preprocessing - handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Convert categorical variables if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:  # One-hot encode if few categories
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df = df.drop(col, axis=1)
        else:  # Otherwise just drop (or could use target encoding)
            df = df.drop(col, axis=1)
    
    # Feature engineering
    if feature_engineering and len(numeric_cols) > 0:
        # Create polynomial features for numerical columns
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_cols])
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
    
    try:
        # SHAP Analysis
        explainer = shap.Explainer(model)
        shap_values = explainer(X[:100])  # Use first 100 samples for speed
        
        plt.figure()
        shap.summary_plot(shap_values, X[:100], feature_names=feature_names, show=False)
        shap_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(shap_path)
        
        # LIME Analysis
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X, 
            feature_names=feature_names,
            class_names=[str(x) for x in np.unique(model.classes_)],
            verbose=False,
            mode='classification'
        )
        
        # Explain a random instance
        exp = explainer.explain_instance(X[0], model.predict_proba, num_features=5)
        lime_path = os.path.join(output_dir, 'lime_explanation.png')
        exp.as_pyplot_figure().savefig(lime_path, bbox_inches='tight')
        plt.close()
        plot_paths.append(lime_path)
        
    except Exception as e:
        print(f"Explainability failed: {str(e)}")
    
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
    
    metrics = None
    explainability_plots = None
    
    try:
        # Load and preprocess data
        df = pd.read_csv(csv_file)
        
        if perform_ml and len(df.columns) > 1:
            try:
                processed_df = preprocess_data(df)
                
                # Assume last column is target for demonstration
                if len(processed_df.columns) > 1:  # Ensure we still have features after preprocessing
                    X = processed_df.iloc[:, :-1].values
                    y = processed_df.iloc[:, -1].values
                    
                    # Convert y to numeric if needed
                    if y.dtype == object:
                        y = pd.factorize(y)[0]
                    
                    # Evaluate baseline model
                    baseline_model = RandomForestClassifier(random_state=42, n_estimators=100)
                    metrics = evaluate_model(X, y, baseline_model)
                    
                    # Generate explainability plots
                    feature_names = processed_df.columns[:-1]
                    explainability_plots = generate_explainability_plots(X, baseline_model, feature_names)
                    
                    wandb.log(metrics)
            except Exception as e:
                print(f"ML analysis failed: {str(e)}")
                wandb.log({"ml_error": str(e)})
        
        # Run the main analysis
        agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"])
        analysis_result = agent.run("""
            You are an expert data analyst. Perform comprehensive analysis including:
            1. Basic statistics and data quality checks
            2. 3 insightful analytical questions about relationships in the data
            3. Visualization of key patterns and correlations
            4. Actionable real-world insights derived from findings
            Generate publication-quality visualizations and save to './figures/'
        """, additional_args={"additional_notes": additional_notes, "source_file": csv_file})
        
    except Exception as e:
        analysis_result = f"Analysis failed: {str(e)}"
    
    execution_time = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 ** 2
    memory_usage = final_memory - initial_memory
    wandb.log({"execution_time_sec": execution_time, "memory_usage_mb": memory_usage})
    
    visuals = [os.path.join('./figures', f) for f in os.listdir('./figures') if f.endswith(('.png', '.jpg', '.jpeg'))]
    for viz in visuals:
        wandb.log({os.path.basename(viz): wandb.Image(viz)})
    
    run.finish()
    return format_analysis_report(analysis_result, visuals, metrics, explainability_plots)

def objective(trial, csv_path):
    try:
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        processed_df = preprocess_data(df)
        
        if len(processed_df.columns) <= 1:
            return 0.0  # No features to work with
        
        X = processed_df.iloc[:, :-1].values
        y = processed_df.iloc[:, -1].values
        
        # Convert y to numeric if needed
        if y.dtype == object:
            y = pd.factorize(y)[0]
        
        # Define hyperparameter space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        # Split data
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
    
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        return 0.0

def tune_hyperparameters(n_trials: int, csv_file):
    try:
        if not csv_file:
            return "Please upload a CSV file first for hyperparameter tuning."
        
        # Save the uploaded file temporarily for Optuna
        temp_path = "temp_optuna_data.csv"
        with open(temp_path, "wb") as f:
            f.write(csv_file.read())
        
        # Verify the data can be loaded
        df = pd.read_csv(temp_path)
        if len(df.columns) <= 1:
            os.remove(temp_path)
            return "Dataset needs at least one feature and one target column."
        
        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, temp_path), n_trials=n_trials)
        
        os.remove(temp_path)
        return f"""
        Best Hyperparameters: {study.best_params}
        Best F1 Score: {study.best_value:.4f}
        """
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
            with gr.Accordion("Hyperparameter Tuning", open=False):
                optuna_trials = gr.Number(label="Number of Trials", value=10, precision=0)
                tune_btn = gr.Button("Optimize Hyperparameters", variant="secondary")
        with gr.Column():
            analysis_output = gr.HTML("""<div style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #2B547E;">Analysis results will appear here...</h2>
                <p>Upload a CSV file and click "Analyze" to begin.</p>
            </div>""")
            optuna_output = gr.Textbox(label="Tuning Results", interactive=False)
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

if __name__ == "__main__":
    demo.launch(debug=True)