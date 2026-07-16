# MLOps-Zombie-Recognition

Data: https://www.kaggle.com/datasets/kingabzpro/zombies-apocalypse/code?datasetId=1074496


## 🧟 Session 2: Enterprise Training Pipeline & Model Registry

In this session, we successfully built and operationalized a robust, end-to-end Machine Learning pipeline for the CDC Zombie Defense System. 

### ⚙️ What We Built
1. **Unified Scikit-Learn Pipeline**: Integrates custom feature engineering, missing value imputation, categorical encoding, and feature scaling alongside our classification model.
2. **MLflow Model Registry Integration**: Automates tracking of training runs, logs performance metrics, and registers the trained model under version control (`CDC_Z_Defense_Pipeline`).
3. **Local Serialization**: Outputs a production-ready serialized pipeline artifact (`zombie_pipeline.pkl`) containing all preprocessing steps and the fitted estimator.
4. **Production Inference Testing**: Developed a standalone testing script (`predict_test.py`) that loads the serialized pipeline, processes raw, un-preprocessed patient data, and outputs highly confident predictions.

### 📊 Performance Metrics
* **Test Accuracy**: `90.00%` (0.9000)
* **Sample Test Output**: 
  * **Status**: `HUMAN 🟢`
  * **Confidence Level**: `84.00%`

### 🏃‍♂️ How to Run

1. **Train and Register the Pipeline**:
   ```bash
   python train_pipeline.py

 2. **Run Production Inference Test**:
    
    python predict_test.py
