# phase6_production/pipeline/streamlit_dashboard.py
# Interactive Streamlit Dashboard for Data Quality Assessment

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from ml_classifier import QualityMLClassifier
from data_loader import LabeledDataLoader
from feature_engineer import QualityFeatureEngineer
from quality_processor import QualityProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Data Quality Assessment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["📊 Dashboard", "🤖 Model Performance", "🔍 Data Analysis", "📈 Predictions", "ℹ️ About"]
)

# Load data in session state
@st.cache_resource
def load_data():
    """Load and process all datasets."""
    try:
        loader = LabeledDataLoader()
        engineer = QualityFeatureEngineer()
        processor = QualityProcessor()
        
        # Load datasets
        titanic = loader.load_titanic()
        ecommerce = loader.load_ecommerce()
        hr = loader.load_hr()
        
        # Extract features
        titanic_features = engineer.extract_all_features(titanic)
        ecommerce_features = engineer.extract_all_features(ecommerce)
        hr_features = engineer.extract_all_features(hr)
        
        # Compute quality scores
        titanic_scored = processor.compute_quality_scores(titanic_features)
        ecommerce_scored = processor.compute_quality_scores(ecommerce_features)
        hr_scored = processor.compute_quality_scores(hr_features)
        
        # ✅ FIXED: Use final_label from ground truth instead of classify_records
        titanic_scored['quality_class'] = titanic_scored['final_label'].map({1: 'HIGH', 0: 'LOW'})
        ecommerce_scored['quality_class'] = ecommerce_scored['final_label'].map({1: 'HIGH', 0: 'LOW'})
        hr_scored['quality_class'] = hr_scored['final_label'].map({1: 'HIGH', 0: 'LOW'})
        
        return {
            'titanic': titanic_scored,
            'ecommerce': ecommerce_scored,
            'hr': hr_scored,
            'combined': pd.concat([titanic_scored, ecommerce_scored, hr_scored], ignore_index=True)
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load trained ML classifier."""
    try:
        classifier = QualityMLClassifier()
        model_path = Path(__file__).parent / "quality_classifier_model.pkl"
        
        if model_path.exists():
            classifier.load_model(str(model_path))
            return classifier
        else:
            st.warning("Trained model not found. Please train the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "📊 Dashboard":
    st.title("📊 Data Quality Assessment Dashboard")
    
    data = load_data()
    if data is None:
        st.stop()
    
    combined_data = data['combined']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(combined_data):,}")
    
    with col2:
        high_quality = len(combined_data[combined_data['quality_class'] == 'HIGH'])
        pct = (high_quality / len(combined_data)) * 100
        st.metric("High Quality", f"{high_quality:,}", f"{pct:.1f}%")
    
    with col3:
        avg_score = combined_data['quality_score'].mean()
        st.metric("Avg Quality Score", f"{avg_score:.1%}")
    
    with col4:
        low_quality = len(combined_data[combined_data['quality_class'] == 'LOW'])
        pct = (low_quality / len(combined_data)) * 100
        st.metric("Low Quality", f"{low_quality:,}", f"{pct:.1f}%")
    
    # Dataset comparison
    st.subheader("Dataset Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        # Dataset size comparison
        datasets_info = {
            'Titanic': len(data['titanic']),
            'E-Commerce': len(data['ecommerce']),
            'HR Analytics': len(data['hr'])
        }
        
        fig_size = go.Figure(data=[
            go.Bar(
                x=list(datasets_info.keys()),
                y=list(datasets_info.values()),
                marker_color=['#667eea', '#764ba2', '#f093fb']
            )
        ])
        fig_size.update_layout(
            title="Records per Dataset",
            xaxis_title="Dataset",
            yaxis_title="Number of Records",
            height=400
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        # Quality distribution pie chart
        quality_dist = combined_data['quality_class'].value_counts()
        
        fig_quality = go.Figure(data=[
            go.Pie(
                labels=quality_dist.index,
                values=quality_dist.values,
                marker_colors=['#28a745', '#dc3545'],
                hole=0.3
            )
        ])
        fig_quality.update_layout(
            title="Quality Classification Distribution",
            height=400
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # Quality score distribution
    st.subheader("Quality Score Distribution")
    
    fig_dist = go.Figure()
    
    for dataset_name, dataset_df in [('Titanic', data['titanic']), 
                                      ('E-Commerce', data['ecommerce']), 
                                      ('HR Analytics', data['hr'])]:
        fig_dist.add_trace(go.Histogram(
            x=dataset_df['quality_score'],
            name=dataset_name,
            opacity=0.7,
            nbinsx=30
        ))
    
    fig_dist.update_layout(
        title="Quality Score Distribution by Dataset",
        xaxis_title="Quality Score",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.title("🤖 ML Classifier Performance")
    
    classifier = load_model()
    if classifier is None or classifier.model is None:
        st.error("Model not trained yet. Please train the model first using ml_classifier.py")
        st.stop()
    
    data = load_data()
    combined_data = data['combined']
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = classifier.get_feature_importance()
    
    features = list(importance.keys())
    scores = list(importance.values())
    
    fig_importance = go.Figure(data=[
        go.Bar(
            y=features,
            x=scores,
            orientation='h',
            marker_color='#667eea'
        )
    ])
    fig_importance.update_layout(
        title="Top Features for Quality Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Training information
    st.subheader("Training Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Training Samples:** {classifier.training_history.get('train_set_size', 'N/A')}")
    
    with col2:
        st.info(f"**Test Samples:** {classifier.training_history.get('test_set_size', 'N/A')}")
    
    with col3:
        st.info(f"**Features Used:** {len(classifier.feature_names)}")

# ============================================================================
# PAGE: DATA ANALYSIS
# ============================================================================

elif page == "🔍 Data Analysis":
    st.title("🔍 Detailed Data Analysis")
    
    data = load_data()
    if data is None:
        st.stop()
    
    # Dataset selector
    dataset_name = st.selectbox("Select Dataset:", ["Titanic", "E-Commerce", "HR Analytics"])
    
    dataset_map = {
        "Titanic": data['titanic'],
        "E-Commerce": data['ecommerce'],
        "HR Analytics": data['hr']
    }
    
    selected_data = dataset_map[dataset_name]
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(selected_data))
    
    with col2:
        high_count = len(selected_data[selected_data['quality_class'] == 'HIGH'])
        st.metric("High Quality", f"{high_count} ({high_count/len(selected_data)*100:.1f}%)")
    
    with col3:
        avg_score = selected_data['quality_score'].mean()
        st.metric("Average Score", f"{avg_score:.1%}")
    
    # Quality metrics breakdown
    st.subheader("Quality Metrics Breakdown")
    
    metric_cols = [col for col in selected_data.columns if col.endswith('_score')]
    metrics_data = selected_data[metric_cols].describe().T
    
    fig_metrics = go.Figure(data=[
        go.Box(
            y=selected_data[col],
            name=col.replace('_score', ''),
            showlegend=True
        )
        for col in metric_cols[:6]  # Show first 6 metrics
    ])
    
    fig_metrics.update_layout(
        title="Distribution of Quality Metrics",
        yaxis_title="Score",
        height=400
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Data table
    st.subheader("Sample Records")
    display_cols = ['quality_score', 'quality_class'] + metric_cols[:5]
    st.dataframe(selected_data[display_cols].head(10), use_container_width=True)

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

elif page == "📈 Predictions":
    st.title("📈 Quality Predictions")
    
    classifier = load_model()
    if classifier is None or classifier.model is None:
        st.error("Model not trained yet.")
        st.stop()
    
    data = load_data()
    if data is None:
        st.stop()
    
    combined_data = data['combined']
    
    # Prediction method selector
    pred_method = st.radio("Prediction Method:", ["Predict on Dataset", "Single Record Prediction"])
    
    if pred_method == "Predict on Dataset":
        dataset_name = st.selectbox("Select Dataset:", ["Titanic", "E-Commerce", "HR Analytics", "All Combined"])
        dataset_map = {
            "Titanic": data['titanic'],
            "E-Commerce": data['ecommerce'],
            "HR Analytics": data['hr'],
            "All Combined": combined_data
        }
        selected_data = dataset_map[dataset_name]
        
        # Make predictions
        predictions = classifier.predict(selected_data)  # Returns [0, 1]
        probabilities = classifier.predict_proba(selected_data)
        
        # Add predictions to dataframe
        results = selected_data.copy()
        # ✅ FIXED: Convert integer predictions [0, 1] to strings ['LOW', 'HIGH']
        results['predicted_quality'] = pd.Series(
            ['HIGH' if pred == 1 else 'LOW' for pred in predictions],
            index=results.index
        )
        results['prediction_confidence'] = probabilities
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            correct = sum(results['predicted_quality'] == results['quality_class'])
            accuracy = correct / len(results) * 100
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
        with col2:
            high_pred = sum(results['predicted_quality'] == 'HIGH')
            st.metric("Predicted HIGH", f"{high_pred} ({high_pred/len(results)*100:.1f}%)")
        with col3:
            avg_conf = results['prediction_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        # ✅ Confusion matrix - Now both are strings!
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            results['quality_class'],
            results['predicted_quality'],
            labels=['HIGH', 'LOW']
    	)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted HIGH', 'Predicted LOW'],
            y=['Actual HIGH', 'Actual LOW'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues'
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Display predictions table
        display_df = results[['quality_score', 'quality_class', 'predicted_quality', 'prediction_confidence']].copy()
        display_df['Match'] = results['predicted_quality'] == results['quality_class']
        
        st.subheader("Detailed Predictions")
        st.dataframe(display_df.head(20), use_container_width=True)
    
    else:  # Single record prediction
        st.subheader("Predict Quality for Single Record")
        
        # Get feature ranges for sliders
        feature_cols = classifier.feature_names
        feature_sliders = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            for feature in feature_cols[:len(feature_cols)//2]:
                min_val = combined_data[feature].min()
                max_val = combined_data[feature].max()
                mean_val = combined_data[feature].mean()
                
                feature_sliders[feature] = st.slider(
                    f"{feature}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=0.01
                )
        
        with col2:
            for feature in feature_cols[len(feature_cols)//2:]:
                min_val = combined_data[feature].min()
                max_val = combined_data[feature].max()
                mean_val = combined_data[feature].mean()
                
                feature_sliders[feature] = st.slider(
                    f"{feature}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    step=0.01
                )
        
        # Create record and predict
        record_df = pd.DataFrame([feature_sliders])
        prediction = classifier.predict(record_df)[0]
        confidence = classifier.predict_proba(record_df)[0]
        
        # Display result
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            color = "green" if prediction == "HIGH" else "red"
            st.markdown(f"""
            <h2 style="color: {color}; text-align: center;">
            Predicted: {prediction} Quality
            </h2>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Data Quality Assessment Framework
    
    This interactive dashboard is part of a comprehensive **Phase 5 ML-based Data Quality Assessment System**.
    
    ### Features
    
    ✅ **Automated Quality Scoring** - 11 quality dimensions analyzed  
    ✅ **ML Classification** - Random Forest predicts HIGH/LOW quality  
    ✅ **Interactive Dashboard** - Real-time visualizations and predictions  
    ✅ **Multi-Dataset Support** - Titanic, E-Commerce, HR Analytics  
    ✅ **Model Explainability** - Feature importance and prediction transparency  
    
    ### Datasets
    
    - **Titanic**: 89 historical records - *94.9% quality (no issues)*
    - **E-Commerce**: 1,000 Brazilian records - *86.5% quality (66.1% duplicates)*
    - **HR Analytics**: 147 HR records - *86.0% quality (70.7% duplicates)*
    
    ### Quality Dimensions
    
    1. **Completeness** - Missing value ratio
    2. **Uniqueness** - Duplicate record detection
    3. **Accuracy** - Data consistency
    4. **Timeliness** - Date freshness
    5. **Validity** - Format compliance
    6. **Consistency** - Cross-field validation
    7. **Conformity** - Schema adherence
    8. **Integrity** - Referential integrity
    9. **Distribution** - Outlier detection
    10. **Entropy** - Data variability
    11. **Distinctiveness** - Cardinality analysis
    
    ### Model Performance
    
    - **Accuracy**: ~92%
    - **Precision**: ~90%
    - **Recall**: ~90%
    - **F1-Score**: ~90%
    
    ### Next Steps
    
    1. Deploy model to production
    2. Set up real-time monitoring
    3. Integrate with data pipelines
    4. Create automated alerts for quality drops
    5. Implement feedback loop for continuous improvement
    
    ---
    
    **Project Status**: ✅ Phase 5 Complete - ML & Dashboard Operational
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("**Phase 5 - ML Model & Dashboard** ✅")
st.sidebar.markdown("*Data Quality Assessment Framework*")