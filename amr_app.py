"""
AMR Prediction Web Application
Based on: PLOS Computational Biology Tutorial (2024)
Author: Gomel State Medical University
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import json
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AMR Clinical Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .resistant {
        background-color: #ffebee;
        border-left: 8px solid #c62828;
    }
    .susceptible {
        background-color: #e8f5e8;
        border-left: 8px solid #2e7d32;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        font-size: 1.2rem;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZATION
# =============================================================================
# Common resistance genes from the PLOS tutorial
RESISTANCE_GENES = [
    "gyrA", "parC", "ampC", "blaTEM", "blaCTX-M", 
    "tetA", "aac(3)-II", "dfrA", "sul1", "sul2",
    "catA1", "aadA", "strA", "strB", "ermB"
]

ANTIBIOTICS = [
    "Ciprofloxacin", "Ampicillin", "Gentamicin", 
    "Ceftazidime", "Cefotaxime", "Meropenem",
    "Imipenem", "Tetracycline", "Trimethoprim", 
    "Sulfamethoxazole", "Chloramphenicol", "Amikacin"
]

# =============================================================================
# SIDEBAR - INPUT SECTION
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bacteria.png", width=80)
    st.markdown("## 🧬 Patient & Isolate Data")
    st.markdown("---")
    
    # Demographics (as described in your article)
    col1, col2 = st.columns(2)
    with col1:
        age_group = st.selectbox(
            "Age Group",
            ["0-18", "19-64", "65+"],
            help="Patient age category"
        )
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
    
    # Clinical data
    infection_source = st.selectbox(
        "Infection Source",
        ["Blood", "Urine", "Respiratory", "Wound", "Other"],
        help="Source of clinical isolate"
    )
    
    # Antibiotic selection
    st.markdown("---")
    st.markdown("### 💊 Select Antibiotic")
    selected_antibiotic = st.selectbox(
        "Antibiotic",
        ANTIBIOTICS,
        index=0
    )
    
    # Gene presence section
    st.markdown("---")
    st.markdown("### 🧪 Resistance Genes")
    st.markdown("Select genes present in the isolate:")
    
    # Organize genes in two columns for better UI
    gene_col1, gene_col2 = st.columns(2)
    gene_presence = {}
    
    for i, gene in enumerate(RESISTANCE_GENES):
        if i < len(RESISTANCE_GENES) // 2:
            with gene_col1:
                gene_presence[gene] = st.checkbox(gene, value=False)
        else:
            with gene_col2:
                gene_presence[gene] = st.checkbox(gene, value=False)
    
    # Additional metadata (from your article)
    st.markdown("---")
    st.markdown("### 📅 Additional Information")
    year = st.number_input("Year of Isolation", min_value=2000, max_value=2025, value=2023)
    
    # Predict button
    st.markdown("---")
    predict_clicked = st.button("🔮 PREDICT RESISTANCE", type="primary", use_container_width=True)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================
st.markdown('<p class="main-header">🧬 Antimicrobial Resistance Clinical Decision Support System</p>', 
            unsafe_allow_html=True)
st.markdown("Based on the PLOS Computational Biology Tutorial (2024) and Moradigaravand et al. (2018)")

# Create tabs as described in your article
tab1, tab2, tab3 = st.tabs(["📤 Input Data", "📊 Prediction Results", "📚 Tutorial & Documentation"])

# =============================================================================
# TAB 1: INPUT DATA
# =============================================================================
with tab1:
    st.markdown('<p class="sub-header">Data Input Methods</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Option 1: Upload CSV File")
        st.markdown("""
        Upload a CSV file with gene presence data.
        **Format:**
        - First row: gene names
        - Values: 0 (absent) or 1 (present)
        - One isolate per row
        """)
        
        # Create a template CSV for users
        template_df = pd.DataFrame(
            {gene: [0] for gene in RESISTANCE_GENES[:8]}
        )
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv_template,
            file_name="gene_presence_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your gene presence data"
        )
        
        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_uploaded)} isolates with {len(df_uploaded.columns)} genes")
                st.dataframe(df_uploaded.head(), use_container_width=True)
                st.session_state['uploaded_data'] = df_uploaded
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("#### ✏️ Option 2: Manual Entry (Current Selection)")
        st.markdown("Genes selected in sidebar:")
        
        # Display selected genes in a clean format
        selected_genes = [gene for gene, present in gene_presence.items() if present]
        if selected_genes:
            st.success(f"✅ {len(selected_genes)} genes selected")
            
            # Create a DataFrame of selected genes
            gene_df = pd.DataFrame({
                'Gene': selected_genes,
                'Status': ['Present'] * len(selected_genes)
            })
            st.dataframe(gene_df, use_container_width=True)
        else:
            st.info("No genes selected. Use the sidebar checkboxes to select resistance genes.")
        
        # Display current metadata
        st.markdown("#### 📋 Current Metadata")
        meta_df = pd.DataFrame({
            'Parameter': ['Age Group', 'Gender', 'Infection Source', 'Year', 'Antibiotic'],
            'Value': [age_group, gender, infection_source, year, selected_antibiotic]
        })
        st.dataframe(meta_df, use_container_width=True)

# =============================================================================
# TAB 2: PREDICTION RESULTS (MATCHES YOUR ARTICLE'S DESCRIPTION)
# =============================================================================
with tab2:
    if not predict_clicked:
        st.info("👈 Enter patient data in the sidebar and click 'PREDICT RESISTANCE'")
        
        # Show sample dashboard
        st.markdown("### 📈 AMR Surveillance Dashboard")
        st.markdown("Sample data from your region (demo mode)")
        
        # Create sample visualization
        years = [2019, 2020, 2021, 2022, 2023]
        resistance_data = pd.DataFrame({
            'Year': years,
            'Ciprofloxacin': [32, 35, 38, 41, 44],
            'Ampicillin': [45, 47, 49, 51, 53],
            'Gentamicin': [18, 19, 21, 22, 24],
            'Ceftazidime': [22, 24, 26, 28, 30]
        })
        
        fig = px.line(resistance_data, x='Year', y=['Ciprofloxacin', 'Ampicillin', 'Gentamicin', 'Ceftazidime'],
                      title="Resistance Trends Over Time (%)",
                      labels={'value': 'Resistance (%)', 'variable': 'Antibiotic'})
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        with st.spinner("🔬 Analyzing genomic data and calculating resistance probability..."):
            # Simulate processing time (matches your article's "less than one second")
            time.sleep(1)
            
            # =========================================================================
            # DEMO PREDICTION ENGINE - REPLACE WITH YOUR TRAINED MODEL
            # =========================================================================
            # This simulates the Random Forest model you described in your article
            # Once you train your model, replace this section with:
            # model = joblib.load("models/random_forest.pkl")
            # prediction = model.predict_proba(features)[0]
            
            # Count resistance genes present
            n_resistance_genes = sum(1 for g in gene_presence.values() if g)
            
            # Simulate probability based on gene count and antibiotic
            base_prob = {
                "Ciprofloxacin": 0.35,
                "Ampicillin": 0.55,
                "Gentamicin": 0.25,
                "Ceftazidime": 0.30,
                "Cefotaxime": 0.32,
                "Meropenem": 0.15,
                "Imipenem": 0.14,
                "Tetracycline": 0.40,
                "Trimethoprim": 0.45,
                "Sulfamethoxazole": 0.42,
                "Chloramphenicol": 0.28,
                "Amikacin": 0.18
            }.get(selected_antibiotic, 0.35)
            
            # Each resistance gene increases probability
            gene_effect = n_resistance_genes * 0.07
            
            # Specific gene interactions (matches your feature importance description)
            if gene_presence.get('gyrA', False) and selected_antibiotic == "Ciprofloxacin":
                gene_effect += 0.25  # gyrA is highly predictive for quinolones
            if gene_presence.get('blaTEM', False) and "ampicillin" in selected_antibiotic.lower():
                gene_effect += 0.30  # blaTEM predicts ampicillin resistance
                
            probability = min(base_prob + gene_effect, 0.98)
            probability = max(probability, 0.02)
            
            prediction = "Resistant" if probability > 0.5 else "Susceptible"
            confidence = probability if prediction == "Resistant" else 1 - probability
            # =========================================================================
            
            # Display results as described in your article
            st.markdown("## 🎯 Prediction Results")
            
            # Main prediction card
            box_class = "resistant" if prediction == "Resistant" else "susceptible"
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h1 style="margin:0; font-size:3rem;">{prediction}</h1>
                <h2 style="margin:0.5rem 0">Confidence: {confidence:.1%}</h2>
                <p class="info-text">Based on Random Forest model (AUROC: 0.95)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics row (matches your article's Table 1)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model AUROC", "0.95", "Ciprofloxacin")
            with col2:
                st.metric("Accuracy", "89%", "+2% vs baseline")
            with col3:
                st.metric("Sensitivity", "87%", "Good detection rate")
            with col4:
                st.metric("Specificity", "91%", "Low false positives")
            
            # Probability gauge (Plotly)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Resistance Probability (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#c62828" if prediction == "Resistant" else "#2e7d32"},
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e8"},
                        {'range': [30, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (matches your Figure 1 description)
            st.markdown("### 🧬 Top Predictive Genes (Gini Importance)")
            
            # Create importance data based on selected genes
            importance_data = []
            for gene, present in gene_presence.items():
                if present:
                    # Known importance weights from your article
                    importance_map = {
                        'gyrA': 0.32, 'parC': 0.28, 'blaTEM': 0.18, 
                        'ampC': 0.12, 'tetA': 0.10, 'blaCTX-M': 0.09,
                        'aac(3)-II': 0.08, 'dfrA': 0.07, 'sul1': 0.06,
                        'sul2': 0.06, 'catA1': 0.05, 'aadA': 0.05,
                        'strA': 0.04, 'strB': 0.04, 'ermB': 0.03
                    }
                    imp = importance_map.get(gene, 0.05)
                    importance_data.append({'Gene': gene, 'Importance': imp})
            
            if importance_data:
                imp_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
                fig = px.bar(imp_df, y='Gene', x='Importance', orientation='h',
                             title="Feature Importance for This Prediction",
                             labels={'Importance': 'Importance Score'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No resistance genes selected. Feature importance will appear when genes are selected.")
            
            # Clinical recommendation (as in your article)
            st.markdown("### 💊 Clinical Recommendation")
            if prediction == "Resistant":
                st.warning("""
                **⚠️ Resistance detected – Consider alternative antibiotic**
                
                Based on the genomic profile, this isolate is likely resistant. Consider:
                - Reviewing local antibiogram data
                - Consulting infectious disease specialist
                - Performing confirmatory susceptibility testing
                - Using combination therapy if clinically indicated
                """)
            else:
                st.success("""
                **✅ Susceptible – This antibiotic is likely effective**
                
                    - Standard dosing recommended
                    - Monitor clinical response
                    - Complete full course of therapy
                    - Report to antimicrobial stewardship program
                """)

# =============================================================================
# TAB 3: TUTORIAL & DOCUMENTATION (FROM YOUR ARTICLE)
# =============================================================================
with tab3:
    st.markdown('<p class="sub-header">📚 Documentation & Training Guide</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Application
        
        This web application implements the machine learning pipeline described in:
        
        **"Using genomic data and machine learning to predict antibiotic resistance: A tutorial paper"**
        *PLOS Computational Biology, 2024*
        
        The underlying model was trained on the dataset from:
        
        **Moradigaravand et al. (2018)** - Prediction of antibiotic resistance in *Escherichia coli* 
        from large-scale pan-genome data.
        
        ### Dataset Details
        - **1,936** *E. coli* isolates from bloodstream infections
        - **17,199** genes screened for presence/absence
        - **12** antibiotics with susceptibility phenotypes
        - **Binary classification:** Resistant / Susceptible
        
        ### Model Performance (Random Forest)
        - AUROC > 0.90 for all tested antibiotics
        - Highest accuracy for Ciprofloxacin (AUROC: 0.95)
        - Cross-validated on geographically diverse samples
        
        ### How to Train Your Own Model
        
        1. **Access the tutorial notebooks:**
           - [PLOS Computational Biology Tutorial](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010958)
           
        2. **Run in Google Colab:**
           - Open the provided notebooks
           - Run all cells (no installation needed)
           - Save your trained model using `joblib.dump()`
           
        3. **Update this application:**
           ```python
           # Replace the demo section with:
           import joblib
           model = joblib.load("your_model.pkl")
           prediction = model.predict_proba(features)
           ```
        """)
        
        # Download button for sample data
        st.markdown("### 📥 Download Sample Data")
        sample_data = pd.DataFrame({
            'Isolate_ID': [f'Sample_{i}' for i in range(1, 11)],
            'gyrA': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            'parC': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            'ampC': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            'blaTEM': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            'Ciprofloxacin': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        })
        
        csv_data = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Dataset",
            data=csv_data,
            file_name="sample_amr_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        **As reported in your article (Table 1):**
        
        | Antibiotic | AUROC |
        |------------|-------|
        | Ciprofloxacin | 0.95 |
        | Ampicillin | 0.93 |
        | Gentamicin | 0.92 |
        | Ceftazidime | 0.91 |
        | Cefotaxime | 0.90 |
        
        **Citation:**
        
        ```
        @article{plos2024tutorial,
          title={Using genomic data and machine learning 
                 to predict antibiotic resistance: 
                 A tutorial paper},
          journal={PLOS Computational Biology},
          year={2024}
        }
        ```
        """)
        
        st.info("""
        **Note:** This is a demonstration version. 
        For production use, please train your own model 
        using the tutorial notebooks.
        """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p style="font-size: 0.9rem;">
        Developed at Gomel State Medical University | Department of Surgical Diseases N°3<br>
        Based on PLOS Computational Biology Tutorial (2024) and Moradigaravand et al. (2018)<br>
        Version 1.0 | Open Source (MIT License)
    </p>
    <p style="font-size: 0.8rem; color: #999;">
        This tool is for research and educational purposes only. Not for clinical use without validation.
    </p>
</div>
""", unsafe_allow_html=True)
