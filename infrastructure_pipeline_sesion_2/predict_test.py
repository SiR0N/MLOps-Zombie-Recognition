"""
CDC Zombie Outbreak Response Unit
Session 2 - Production Inference Test Script
Description: This script loads the serialized pipeline artifact (.pkl) 
             and performs an inference on raw, un-preprocessed test cases.
"""

import os
import joblib
import pandas as pd

def main():
    print("🔬 [CDC INFRASTRUCTURE] Initializing Production Inference Test...")

    # 1. LOAD THE SERIALIZED PIPELINE
    model_path = "zombie_pipeline.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Serialized pipeline not found at '{model_path}'. Please run 'train_pipeline.py' first.")
        
    print(f"📦 Loading pipeline from '{model_path}'...")
    pipeline = joblib.load(model_path)
    print("✅ Pipeline loaded successfully into memory.")

    # 2. CREATE A RAW PATIENT TEST CASE (Simulating real-time API request)
    # Notice how we pass raw data: missing values, text strings, etc.
    # The pipeline will handle imputation and encoding automatically!
    print("📋 Mocking a raw incoming patient data from the field...")
    raw_patient_data = {
        'age': [35.0],              # Numerical
        'sex': ['Female'],          # Categorical
        'rurality': ['Rural'],      # Categorical
        'household': [4.0],         # Numerical
        'water': [2.0],             # Numerical
        'food': ['Food sector'],    # Categorical
        'medication': [None],       # Categorical with MISSING value (None)
        'tools': ['no'],            # Categorical
        'firstaid': ['yes'],        # Categorical
        'sanitation': ['no'],       # Categorical
        'clothing': ['yes']         # Categorical
    }

    # Convert to pandas DataFrame
    test_df = pd.DataFrame(raw_patient_data)

    # 3. RUN INFERENCE THROUGH THE PIPELINE
    print("🔮 Running data through the unified Pipeline...")
    prediction = pipeline.predict(test_df)
    probability = pipeline.predict_proba(test_df)

    # 4. REPORT RESULTS
    result = "ZOMBIE 🧟‍♂️" if prediction[0] == 1 else "HUMAN 🟢"
    confidence = probability[0][prediction[0]] * 100

    print("\n" + "="*50)
    print("🚨 [CDC CLASSIFICATION REPORT] 🚨")
    print(f"Status: {result}")
    print(f"Confidence Level: {confidence:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()