import streamlit as st
import pandas as pd
import joblib
import os

# Configuración de la interfaz
st.set_page_config(
    page_title="CDC Survival & Zombie Diagnostic Portal",
    page_icon="🧟",
    layout="wide"
)

# Cabecera de la aplicación
st.title("🧟 CDC Survival & Zombie Diagnostic - Session 2")
st.markdown("""
Welcome to the tactical diagnostic application. This system utilizes the **Scikit-Learn Pipeline** trained during **Session 2** to predict whether a survivor is classified as a **Human** or a **Zombie** based on their survival parameters, supplies, and demographic features.
""")

# Crear las dos pestañas interactivas
tab1, tab2 = st.tabs(["🎯 Live Diagnostic Screening", "⚙️ Pipeline Architecture & Code"])

# --- PESTAÑA 1: DIAGNÓSTICO EN VIVO ---
with tab1:
    st.header("Field Screening Form")
    st.write("Modify the demographic variables and supplies of the survivor to request a pipeline prediction.")
    
    # Creamos 3 columnas para organizar un formulario limpio y estético
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographic Info")
        age = st.slider("Age (años)", 0, 100, 25)
        sex = st.selectbox("Sex", ["Female", "Male"])
        rurality = st.selectbox("Rurality", ["Rural", "Suburban", "Urban"])
        household = st.number_input("Household Members", min_value=1, max_value=20, value=2)
        
    with col2:
        st.subheader("💧 Survival Supplies (Numeric)")
        water = st.number_input("Water (Litros/Galones)", min_value=0, max_value=100, value=8)
        st.write("")
        st.subheader("🍎 Vital Supplies (Categorical)")
        food = st.selectbox("Food Status", ["Food", "No food"])
        medication = st.selectbox("Medication Status", ["Medication", "No medication"])

    with col3:
        st.subheader("🎒 Equipment & Items")
        tools = st.selectbox("Tools", ["tools", "No tools"])
        firstaid = st.selectbox("First Aid", ["First aid supplies", "No first aid supplies"])
        sanitation = st.selectbox("Sanitation", ["Sanitation", "No sanitation"])
        clothing = st.selectbox("Clothing", ["Clothing", "NA"])
        documents = st.selectbox("Documents", ["Documents", "NA"])

    # Botón para accionar la predicción
    st.markdown("---")
    if st.button("Run Diagnostic Scan 🧠", type="primary", use_container_width=True):
        # 1. Empaquetar los datos del formulario con los nombres EXACTOS de las columnas de tu zombies.csv
        # Excluimos "zombieid" y "zombie" (el target) tal como hace tu pipeline al entrenar.
        raw_input = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'rurality': rurality,
            'household': household,
            'water': water,
            'food': food,
            'medication': medication,
            'tools': tools,
            'firstaid': firstaid,
            'sanitation': sanitation,
            'clothing': clothing,
            'documents': documents
        }])
        
        # 2. Definir la ruta de tu pipeline de la Sesión 2
        # (Ajusta la subcarpeta si cambia de nombre, en tus logs locales era 'infrastructure_pipeline_sesion_2')
        pipeline_path = os.path.join("infrastructure_pipeline_sesion_2", "zombie_pipeline.pkl")
        
        if os.path.exists(pipeline_path):
            try:
                # Cargar el modelo entrenado
                pipeline = joblib.load(pipeline_path)
                
                # Ejecutar predicción
                prediction = pipeline.predict(raw_input)[0]
                probabilities = pipeline.predict_proba(raw_input)[0]
                classes = pipeline.classes_
                
                # Mostrar el veredicto con formato de alerta visual
                st.markdown("### 🔍 Pipeline Veredict:")
                
                # Comprobamos la clase resultante
                is_zombie = str(prediction).lower() == "zombie" or prediction == 1
                
                # Encontrar el índice de probabilidad correspondiente al resultado
                # (Suele ser 0 para Human y 1 para Zombie, pero nos aseguramos buscando en classes)
                pred_idx = list(classes).index(prediction) if prediction in classes else 0
                confidence = probabilities[pred_idx] * 100
                
                if not is_zombie:
                    st.success(f"🟢 **HUMAN** - Diagnostic confirmation. This survivor shows normal biological signs. (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"🚨 **ZOMBIE** - Emergency quarantine protocols activated! Contamination detected. (Confidence: {confidence:.2f}%)")
                
                # Pequeño panel con el desglose de probabilidades
                with st.expander("Show Detailed Pipeline Probabilities"):
                    prob_df = pd.DataFrame([probabilities], columns=classes)
                    st.dataframe(prob_df)
                    
            except Exception as e:
                st.error(f"Error executing pipeline inference: {e}")
                st.info("Ensure the features in 'app.py' match the transformers defined in 'train_pipeline.py'.")
        else:
            st.warning(f"Serialized pipeline file not found at `{pipeline_path}`. Please run 'python train_pipeline.py' first to generate it!")

# --- PESTAÑA 2: EXPLICACIÓN DEL PIPELINE ---
with tab2:
    st.header("Session 2 Pipeline Details")
    st.markdown("""
    When raw survivor data is entered in the UI, it passes through an automated **Scikit-Learn Pipeline** before classification:
    """)
    
    col_step1, col_step2 = st.columns(2)
    
    with col_step1:
        st.subheader("1. Data Preprocessing & Pipeline")
        st.markdown("""
        * **Numerical Features** (`age`, `household`, `water`): Imputed using `SimpleImputer(strategy='median')` to ensure no missing metrics break the system, followed by scaling.
        * **Categorical Features** (`sex`, `rurality`, etc.): Encoded using `OneHotEncoder(handle_unknown='ignore')` to safely translate labels to binary representation.
        """)
        st.code("""
# Pipeline Preprocessing Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
        """, language="python")

    with col_step2:
        st.subheader("2. Tracking & MLflow Registration")
        st.markdown("""
        * **No Data Leakage**: Pipelines guarantee that data transformations are only fit on training folds and never 'leak' test parameters.
        * **MLflow Run Tracking**: During execution, parameters and test accuracy are securely pushed to **MLflow**, creating an official model register.
        """)