"""
CDC Zombie Outbreak Response Unit
MLOps Infrastructure Pipeline (Session 2)
Description: This script automates data preprocessing and model wrapping into a 
             single reproducible scikit-learn Pipeline, then registers it in MLflow.
"""

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def main():
    print("🚀 [CDC INFRASTRUCTURE] Starting Enterprise Pipeline Training...")

    # 1. LOAD DATASET
    # Since we are inside the 'infrastructure_pipeline_sesion_2' folder,
    # we look for 'zombies.csv' in the parent directory (root).
    data_path = "../zombies.csv"
    
    if not os.path.exists(data_path):
        # Fallback in case the script is run from the root directory
        data_path = "zombies.csv"
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Could not find 'zombies.csv' at {data_path}. Please make sure the dataset is in the repository.")
        
    df = pd.read_csv(data_path)
    print(f"📊 Dataset loaded successfully from {data_path}. Shape: {df.shape}")

    # 2. DEFINE TARGET & FEATURES
    # Map target: Zombie -> 1, Human -> 0
    df['target'] = df['zombie'].apply(lambda x: 1 if x == 'Zombie' else 0)
    
    # Drop irrelevant columns and target
    X = df.drop(columns=['zombieid', 'zombie', 'target'])
    y = df['target']

    # Define numerical and categorical columns for preprocessing
    num_cols = ['age', 'household', 'water']
    cat_cols = ['sex', 'rurality', 'food', 'medication', 'tools', 'firstaid', 'sanitation', 'clothing']

    # 3. BUILD PREPROCESSING PIPELINES
    print("⚙️ Building Feature Engineering & Preprocessing pipelines...")
    
    # Numerical Pipeline: Impute missing values with median, then scale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: Impute missing values with 'no', then encode
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='no')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # 4. WRAP PREPROCESSOR + ESTIMATOR INTO A SINGLE PIPELINE
    # Wrap preprocessor and model together
    zombie_defense_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. SPLIT DATA FOR VALIDATION
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. MLFLOW TRACKING & REGISTRATION
    # Allow filesystem tracking backend
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Fix Windows space-in-path issue by setting a clean tracking URI in C:\temp
    local_mlflow_path = "C:/temp/mlflow_runs"
    os.makedirs(local_mlflow_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{local_mlflow_path}")
    
    mlflow.set_experiment("zombie_apocalypse_defense")
    
    with mlflow.start_run(run_name="enterprise_infrastructure_v1") as run:
        print("🏋️ Training the complete Machine Learning Pipeline...")
        
        # Train and assign the fully fitted pipeline
        fitted_pipeline = zombie_defense_pipeline.fit(X_train, y_train)

        # Evaluate the unified pipeline
        test_accuracy = fitted_pipeline.score(X_test, y_test)
        print(f"🎯 Evaluation Complete. Test Accuracy: {test_accuracy:.4f}")
        
        # Log metrics in MLflow
        mlflow.log_metric("accuracy", test_accuracy)

        # Log and Register the whole pipeline artifact in MLflow Model Registry
        print("💾 Registering the unified pipeline to MLflow Model Registry...")
        mlflow.sklearn.log_model(
            sk_model=fitted_pipeline,
            artifact_path="zombie_defense_pipeline",
            registered_model_name="CDC_Z_Defense_Pipeline",
            skops_trusted_types=["numpy.dtype"]
        )

    # 7. SERIALIZATION FOR LOCAL DEPLOYMENT (Required for Session 3)
    # Save physical .pkl file inside the session folder
    model_filename = "zombie_pipeline.pkl"
    joblib.dump(zombie_defense_pipeline, "zombie_pipeline.pkl")
    print(f"💾 Pipeline successfully serialized locally to '{model_filename}'")

    print("✅ [CDC INFRASTRUCTURE] Training and registration completed successfully.")

if __name__ == "__main__":
    main()