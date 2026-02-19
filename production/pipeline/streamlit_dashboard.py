# phase6_production/pipeline/streamlit_dashboard.py
# -*- coding: utf-8 -*-
# Interactive Streamlit Dashboard for Data Quality Assessment

import json                                        # ← ADDED (only new import)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Data Quality Assessment Dashboard",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Dashboard", "Model Performance", "Data Analysis", "Predictions",
     "Upload & Profile", "About"]
)

# ============================================================================
# DATA & MODEL LOADERS (cached)
# ============================================================================

@st.cache_resource
def load_data():
    """Load and process all three ground-truth datasets."""
    try:
        loader    = LabeledDataLoader()
        engineer  = QualityFeatureEngineer()
        processor = QualityProcessor()

        titanic   = loader.load_titanic()
        ecommerce = loader.load_ecommerce()
        hr        = loader.load_hr()

        titanic_features   = engineer.extract_all_features(titanic)
        ecommerce_features = engineer.extract_all_features(ecommerce)
        hr_features        = engineer.extract_all_features(hr)

        titanic_scored   = processor.compute_quality_scores(titanic_features)
        ecommerce_scored = processor.compute_quality_scores(ecommerce_features)
        hr_scored        = processor.compute_quality_scores(hr_features)

        # Map ground-truth binary labels to HIGH/LOW string labels
        for df_scored, df_orig in [
            (titanic_scored,   titanic),
            (ecommerce_scored, ecommerce),
            (hr_scored,        hr),
        ]:
            df_scored['quality_class'] = processor.extract_labels(df_orig).map(
                {1: 'HIGH', 0: 'LOW'}
            )

        combined = pd.concat(
            [titanic_scored, ecommerce_scored, hr_scored],
            ignore_index=True
        )
        return {
            'titanic':   titanic_scored,
            'ecommerce': ecommerce_scored,
            'hr':        hr_scored,
            'combined':  combined,
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model():
    """Load trained ML classifier from disk."""
    try:
        classifier = QualityMLClassifier()
        model_path = Path(__file__).parent / "quality_classifier_model.pkl"
        if model_path.exists():
            classifier.load_model(str(model_path))
            return classifier
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def train_and_save_model(data: dict) -> QualityMLClassifier:
    """Train model on combined dataset and save to disk."""
    processor     = QualityProcessor()
    loader        = LabeledDataLoader()
    combined      = data['combined'].copy()

    titanic   = loader.load_titanic()
    ecommerce = loader.load_ecommerce()
    hr        = loader.load_hr()

    combined_orig = pd.concat([titanic, ecommerce, hr], ignore_index=True)
    combined['final_label'] = processor.extract_labels(combined_orig)

    classifier = QualityMLClassifier(random_state=42)
    with st.spinner("Training Random Forest classifier (100 trees, max_depth=15)..."):
        metrics = classifier.train(combined, test_size=0.3, cv_folds=5, target_col='final_label')

    model_path = Path(__file__).parent / "quality_classifier_model.pkl"
    classifier.save_model(str(model_path))
    st.success(
        f"Model trained! "
        f"Accuracy: {metrics['test_accuracy']:.1%} | "
        f"Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f} | "
        f"ROC-AUC: {metrics['roc_auc']:.3f}"
    )
    st.cache_resource.clear()
    return classifier


# ← ADDED: one helper to read phase4_results.json for live H1 status
def load_phase4_results() -> dict:
    """Load live Phase 4 results from phase4_results.json."""
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "phase4_results.json",
        Path(__file__).parent.parent       / "data" / "phase4_results.json",
        Path(__file__).parent              / "phase4_results.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.title("Data Quality Assessment Dashboard")
    st.caption("Adaptive ML Framework - University of Liverpool MSc Data Science & AI")

    data = load_data()
    if data is None:
        st.stop()

    combined_data = data['combined']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(combined_data):,}")
    with col2:
        high_quality = (combined_data['quality_class'] == 'HIGH').sum()
        pct_high = high_quality / len(combined_data) * 100
        st.metric("High Quality", f"{high_quality:,}", f"{pct_high:.1f}%")
    with col3:
        avg_score = combined_data['quality_score'].mean()
        st.metric("Avg Quality Score", f"{avg_score:.1%}")
    with col4:
        low_quality = (combined_data['quality_class'] == 'LOW').sum()
        pct_low = low_quality / len(combined_data) * 100
        st.metric("Low Quality", f"{low_quality:,}", f"{pct_low:.1f}%")

    st.divider()
    st.subheader("Dataset Comparison")
    col1, col2 = st.columns(2)

    with col1:
        datasets_info = {
            'Titanic':      len(data['titanic']),
            'E-Commerce':   len(data['ecommerce']),
            'HR Analytics': len(data['hr']),
        }
        fig_size = go.Figure(data=[go.Bar(
            x=list(datasets_info.keys()),
            y=list(datasets_info.values()),
            marker_color=['#002B5C', '#00838F', '#B3D9E3'],
            text=list(datasets_info.values()),
            textposition='auto',
        )])
        fig_size.update_layout(title="Records per Dataset", xaxis_title="Dataset",
                               yaxis_title="Number of Records", height=380)
        st.plotly_chart(fig_size, use_container_width=True)

    with col2:
        quality_dist = combined_data['quality_class'].value_counts()
        fig_quality = go.Figure(data=[go.Pie(
            labels=quality_dist.index,
            values=quality_dist.values,
            marker_colors=['#002B5C', '#00838F'],
            hole=0.35,
            textinfo='label+percent',
        )])
        fig_quality.update_layout(title="Quality Class Distribution (All Datasets)", height=380)
        st.plotly_chart(fig_quality, use_container_width=True)

    st.subheader("Quality Score Distribution by Dataset")
    fig_dist = go.Figure()
    for dname, dkey, colour in [
        ('Titanic',      'titanic',   '#002B5C'),
        ('E-Commerce',   'ecommerce', '#00838F'),
        ('HR Analytics', 'hr',        '#B3D9E3'),
    ]:
        fig_dist.add_trace(go.Histogram(
            x=data[dkey]['quality_score'], name=dname,
            opacity=0.7, nbinsx=30, marker_color=colour,
        ))
    fig_dist.update_layout(xaxis_title="Quality Score", yaxis_title="Frequency",
                           barmode='overlay', height=380)
    st.plotly_chart(fig_dist, use_container_width=True)


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

elif page == "Model Performance":
    st.title("ML Classifier Performance")

    data       = load_data()
    classifier = load_model()

    if classifier is None or classifier.model is None:
        st.warning("No trained model found (quality_classifier_model.pkl missing).")
        if data is not None:
            if st.button("Train Model Now", type="primary"):
                classifier = train_and_save_model(data)
                st.rerun()
        else:
            st.error("Cannot train: data loading failed.")
        st.stop()

    stored_metrics = classifier.training_history.get('metrics', {})

    st.subheader("Performance Metrics (Test Set)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Test Accuracy",  f"{stored_metrics.get('test_accuracy', 0):.1%}")
    with col2:
        st.metric("Precision",      f"{stored_metrics.get('precision', 0):.3f}", "target >= 0.80")
    with col3:
        st.metric("Recall",         f"{stored_metrics.get('recall', 0):.3f}",    "target >= 0.75")
    with col4:
        st.metric("F1-Score",       f"{stored_metrics.get('f1', 0):.3f}")
    with col5:
        roc_val = stored_metrics.get('roc_auc', None)
        st.metric("ROC-AUC", f"{roc_val:.3f}" if roc_val is not None else "N/A")

    st.divider()

    st.subheader("Cross-Validation Results (5-Fold)")
    cv_mean   = stored_metrics.get('cv_mean', None)
    cv_std    = stored_metrics.get('cv_std',  None)
    cv_scores = stored_metrics.get('cv_scores', [])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CV Mean Accuracy", f"{cv_mean:.4f}" if cv_mean else "N/A")
    with col2:
        st.metric("CV Std Dev",       f"+/- {cv_std:.4f}" if cv_std else "N/A")
    with col3:
        st.metric("Training Size",    f"{classifier.training_history.get('train_set_size', 'N/A')} records")

    if cv_scores:
        fig_cv = go.Figure(data=[go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker_color='#00838F',
            text=[f"{s:.3f}" for s in cv_scores],
            textposition='auto',
        )])
        fig_cv.add_hline(y=cv_mean, line_dash="dash", line_color="#002B5C",
                         annotation_text=f"Mean: {cv_mean:.3f}")
        fig_cv.update_layout(title="Cross-Validation Accuracy per Fold",
                             yaxis_title="Accuracy", yaxis_range=[0.8, 1.0], height=350)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.divider()

    st.subheader("Confusion Matrix")
    cm_data = stored_metrics.get('confusion_matrix', None)
    if cm_data and len(cm_data) == 2:
        cm_array = np.array(cm_data)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=['Predicted HIGH', 'Predicted LOW'],
            y=['Actual HIGH',    'Actual LOW'],
            text=cm_array,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 20},
            colorscale='Blues',
        ))
        fig_cm.update_layout(title="Confusion Matrix (Test Set)",
                             xaxis_title="Predicted Label", yaxis_title="Actual Label", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

        if cm_array.shape == (2, 2):
            TP, FN, FP, TN = int(cm_array[0][0]), int(cm_array[0][1]), int(cm_array[1][0]), int(cm_array[1][1])
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("True Positive (TP)",  TP)
            with col2: st.metric("False Negative (FN)", FN)
            with col3: st.metric("False Positive (FP)", FP)
            with col4: st.metric("True Negative (TN)",  TN)
    else:
        st.info("Confusion matrix not available. Retrain model to generate.")

    st.divider()

    st.subheader("Feature Importance")
    importance = classifier.get_feature_importance()
    features   = list(importance.keys())
    scores     = list(importance.values())

    fig_importance = go.Figure(data=[go.Bar(
        y=features, x=scores, orientation='h',
        marker_color=['#002B5C' if s == max(scores) else '#00838F' for s in scores],
        text=[f"{s:.4f}" for s in scores], textposition='auto',
    )])
    fig_importance.update_layout(
        title="Feature Importance (Random Forest - all 11 features)",
        xaxis_title="Importance Score", yaxis_title="Feature",
        height=480, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Model Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.info("**Algorithm:** Random Forest")
    with col2: st.info("**Trees:** 100  |  **Max Depth:** 15")
    with col3: st.info("**Train Split:** 70%  |  **Test:** 30%")
    with col4: st.info(f"**CV Folds:** {classifier.training_history.get('cv_folds', 5)}")


# ============================================================================
# PAGE: DATA ANALYSIS
# ============================================================================

elif page == "Data Analysis":
    st.title("Detailed Data Analysis")

    data = load_data()
    if data is None:
        st.stop()

    dataset_name = st.selectbox("Select Dataset:", ["Titanic", "E-Commerce", "HR Analytics"])
    dataset_map  = {"Titanic": data['titanic'], "E-Commerce": data['ecommerce'], "HR Analytics": data['hr']}
    selected_data = dataset_map[dataset_name]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", len(selected_data))
    with col2:
        high_count = (selected_data['quality_class'] == 'HIGH').sum()
        st.metric("High Quality", f"{high_count} ({high_count/len(selected_data)*100:.1f}%)")
    with col3:
        low_count = (selected_data['quality_class'] == 'LOW').sum()
        st.metric("Low Quality",  f"{low_count} ({low_count/len(selected_data)*100:.1f}%)")
    with col4:
        st.metric("Average Quality Score", f"{selected_data['quality_score'].mean():.1%}")

    st.divider()

    metric_cols          = [col for col in selected_data.columns if col.endswith('_score')]
    feature_cols_display = [c for c in metric_cols if c != 'quality_score'][:6]

    st.subheader("Quality Feature Score Distributions")
    fig_metrics = go.Figure()
    for col in feature_cols_display:
        fig_metrics.add_trace(go.Box(
            y=selected_data[col],
            name=col.replace('_score', '').replace('_', ' ').title(),
        ))
    fig_metrics.update_layout(title=f"Quality Feature Distributions - {dataset_name}",
                              yaxis_title="Score (0-1)", height=420)
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.subheader("Mean Feature Scores")
    means = selected_data[feature_cols_display].mean().sort_values()
    fig_means = go.Figure(data=[go.Bar(
        x=means.values,
        y=[c.replace('_score', '').replace('_', ' ').title() for c in means.index],
        orientation='h', marker_color='#00838F',
        text=[f"{v:.3f}" for v in means.values], textposition='auto',
    )])
    fig_means.update_layout(xaxis_title="Mean Score", xaxis_range=[0, 1], height=360)
    st.plotly_chart(fig_means, use_container_width=True)

    st.subheader("Sample Records")
    display_cols = ['quality_score', 'quality_class'] + metric_cols[:5]
    st.dataframe(selected_data[display_cols].head(15), use_container_width=True)


# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================

elif page == "Predictions":
    st.title("Quality Predictions")

    classifier = load_model()
    data       = load_data()

    if classifier is None or classifier.model is None:
        st.error("Model not trained yet. Visit Model Performance page to train.")
        st.stop()
    if data is None:
        st.stop()

    combined_data = data['combined']
    pred_method   = st.radio("Prediction Method:", ["Predict on Dataset", "Single Record Prediction"])

    if pred_method == "Predict on Dataset":
        dataset_name = st.selectbox("Select Dataset:",
                                    ["Titanic", "E-Commerce", "HR Analytics", "All Combined"])
        dataset_map = {
            "Titanic":      data['titanic'],
            "E-Commerce":   data['ecommerce'],
            "HR Analytics": data['hr'],
            "All Combined": combined_data,
        }
        selected_data = dataset_map[dataset_name]

        predictions   = classifier.predict(selected_data)      # always strings HIGH/LOW
        probabilities = classifier.predict_proba(selected_data)

        results = selected_data.copy()
        results['predicted_quality']    = predictions
        results['prediction_confidence'] = probabilities

        st.subheader("Prediction Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            correct  = (results['predicted_quality'] == results['quality_class']).sum()
            accuracy = correct / len(results) * 100
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
        with col2:
            high_pred = (results['predicted_quality'] == 'HIGH').sum()
            st.metric("Predicted HIGH", f"{high_pred} ({high_pred/len(results)*100:.1f}%)")
        with col3:
            low_pred = (results['predicted_quality'] == 'LOW').sum()
            st.metric("Predicted LOW",  f"{low_pred} ({low_pred/len(results)*100:.1f}%)")
        with col4:
            st.metric("Avg Confidence", f"{results['prediction_confidence'].mean():.1%}")

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            results['quality_class'],
            results['predicted_quality'],
            labels=['HIGH', 'LOW']
        )
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted HIGH', 'Predicted LOW'],
            y=['Actual HIGH',    'Actual LOW'],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 20},
            colorscale='Blues',
        ))
        fig_cm.update_layout(title=f"Confusion Matrix - {dataset_name}", height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Detailed Predictions (first 20 records)")
        display_df = results[['quality_score', 'quality_class', 'predicted_quality',
                               'prediction_confidence']].copy()
        display_df['Match'] = results['predicted_quality'] == results['quality_class']
        st.dataframe(display_df.head(20), use_container_width=True)

    else:
        # ====================================================================
        # Single Record Prediction
        # ====================================================================
        st.subheader("Predict Quality for a Single Record")
        st.caption("All features are quality scores in [0.0, 1.0]. "
                   "Adjust sliders to simulate a record and see the prediction.")

        feature_cols    = classifier.feature_names
        feature_sliders = {}
        half = len(feature_cols) // 2

        col1, col2 = st.columns(2)

        with col1:
            for feature in feature_cols[:half]:
                mean_val = float(combined_data[feature].mean())
                mean_val = max(0.0, min(1.0, mean_val))  # clamp to [0, 1]
                # Dynamic min/max crashes when min==max (e.g. constant 1.0 features).
                feature_sliders[feature] = st.slider(
                    feature.replace('_score', '').replace('_', ' ').title(),
                    min_value=0.0, max_value=1.0,
                    value=mean_val, step=0.01
                )

        with col2:
            for feature in feature_cols[half:]:
                mean_val = float(combined_data[feature].mean())
                mean_val = max(0.0, min(1.0, mean_val))  # clamp to [0, 1]
                feature_sliders[feature] = st.slider(
                    feature.replace('_score', '').replace('_', ' ').title(),
                    min_value=0.0, max_value=1.0,
                    value=mean_val, step=0.01
                )

        record_df  = pd.DataFrame([feature_sliders])
        prediction = classifier.predict(record_df)[0]
        confidence = classifier.predict_proba(record_df)[0]

        st.divider()
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            icon = "HIGH QUALITY" if prediction == "HIGH" else "LOW QUALITY"
            colour = "green" if prediction == "HIGH" else "red"
            st.markdown(
                f'<h2 style="color:{colour};">{icon}</h2>',
                unsafe_allow_html=True
            )
            st.metric("Confidence", f"{confidence:.1%}")
        with col2:
            st.progress(float(confidence))
            st.caption(f"P(HIGH quality) = {confidence:.4f}")
            if prediction == "HIGH":
                st.success("This record meets quality thresholds.")
            else:
                st.warning("This record falls below quality thresholds.")


# ============================================================================
# PAGE: UPLOAD & PROFILE  (Spec Component 5: File upload widget)
# ============================================================================

elif page == "Upload & Profile":
    st.title("Upload & Profile New Data")
    st.caption(
        "Upload a CSV file to compute quality features, scores, and "
        "get an ML quality prediction — no pre-labeling required."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            import io
            upload_df = pd.read_csv(io.BytesIO(uploaded_file.read()), encoding="utf-8")
            st.success(f"Loaded **{len(upload_df):,}** rows × **{len(upload_df.columns)}** columns")

            # --- Feature extraction ---
            engineer  = QualityFeatureEngineer()
            processor = QualityProcessor()

            with st.spinner("Extracting quality features..."):
                features_df = engineer.extract_all_features(upload_df)
                scored_df   = processor.compute_quality_scores(features_df)
                scored_df   = processor.classify_records(scored_df)

            # --- Summary metrics ---
            st.subheader("Quality Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(scored_df):,}")
            with col2:
                avg_q = scored_df['quality_score'].mean()
                st.metric("Avg Quality Score", f"{avg_q:.1%}")
            with col3:
                high_n = (scored_df['quality_class'] == 'HIGH').sum()
                st.metric("HIGH Quality", f"{high_n:,} ({100*high_n/len(scored_df):.1f}%)")
            with col4:
                low_n = (scored_df['quality_class'] == 'LOW').sum()
                st.metric("LOW Quality", f"{low_n:,} ({100*low_n/len(scored_df):.1f}%)")

            # --- Quality score distribution ---
            st.subheader("Quality Score Distribution")
            fig_hist = go.Figure(data=[go.Histogram(
                x=scored_df['quality_score'], nbinsx=30,
                marker_color='#00838F', opacity=0.8
            )])
            fig_hist.update_layout(
                xaxis_title="Quality Score", yaxis_title="Frequency", height=350
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- ML prediction (if model available) ---
            classifier = load_model()
            if classifier is not None and classifier.model is not None:
                st.subheader("ML Quality Predictions")
                with st.spinner("Running classifier..."):
                    predictions   = classifier.predict(scored_df)
                    probabilities = classifier.predict_proba(scored_df)
                    scored_df['ml_prediction'] = predictions
                    scored_df['ml_confidence'] = probabilities

                col1, col2 = st.columns(2)
                with col1:
                    pred_high = (scored_df['ml_prediction'] == 'HIGH').sum()
                    st.metric("Predicted HIGH", f"{pred_high:,}")
                with col2:
                    st.metric("Avg Confidence", f"{scored_df['ml_confidence'].mean():.1%}")
            else:
                st.info("Train a model on the Model Performance page to enable ML predictions.")

            # --- Quality report ---
            st.subheader("Quality Report")
            report = processor.generate_quality_report(scored_df, uploaded_file.name)
            if report['recommendations']:
                st.warning("**Recommendations:**")
                for rec in report['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.success("No significant quality issues detected.")

            # --- Feature scores table ---
            st.subheader("Feature Scores (first 20 records)")
            score_cols   = [c for c in scored_df.columns if c.endswith('_score')]
            display_cols = ['quality_score', 'quality_class'] + score_cols[:6]
            st.dataframe(scored_df[display_cols].head(20), use_container_width=True)

            # --- Issues breakdown ---
            issues     = processor.identify_issues(scored_df)
            has_issues = any(isinstance(v, dict) and 'percentage' in v for v in issues.values())
            if has_issues:
                st.subheader("Identified Issues")
                for issue_type, details in issues.items():
                    if isinstance(details, dict) and 'percentage' in details:
                        st.markdown(
                            f"- **{issue_type.upper()}**: {details['percentage']:.1f}% "
                            f"of records affected ({details['affected_rows']} rows)"
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            logger.error(f"Upload processing error: {e}")
    else:
        st.info("📂 Upload a CSV file above to begin quality profiling.")


# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":
    st.title("About This System")

    # ── Static markdown (unchanged) ──────────────────────────────────
    st.markdown("""
## Adaptive ML Framework for Automated Data Quality Assessment
### in Heterogeneous Big Data Repositories

**Student:** Lam Chit Wui (27806)
**Programme:** MSc Data Science & AI
**Institution:** University of Liverpool, Department of Computer Science
**Academic Year:** 2025-2026

---
""")

    # ── Research Hypotheses & Status — UPDATED to use live phase4 data ──
    st.markdown("## Research Hypotheses & Status")

    _p4    = load_phase4_results()
    _rf    = _p4.get("rf_test", {})
    _prec  = _rf.get("precision", 0)
    _rec   = _rf.get("recall",    0)
    _f1    = _rf.get("f1_score",  0)
    _h1_ok = _rf.get("h1_pass",   False)

    if _h1_ok:
        _h1_status = f"ACHIEVED  (P={_prec:.3f}  R={_rec:.3f}  F1={_f1:.3f})"
        _h1_color  = "green"
    else:
        _h1_status = f"IN PROGRESS  (P={_prec:.3f}  R={_rec:.3f})"
        _h1_color  = "orange"

    _hypotheses = [
        {
            "id":   "H1",
            "desc": "ML features classify data quality with Precision >= 0.80, Recall >= 0.75",
            "status": _h1_status,
            "color":  _h1_color,
        },
        {
            "id":   "H2",
            "desc": "Framework processes faster than Apache Griffin baseline",
            "status": "IN PROGRESS (March 2026)",
            "color":  "orange",
        },
        {
            "id":   "H3",
            "desc": "Dashboard improves remediation rating >= 3.5/5",
            "status": "ON TRACK (User study March 2026)",
            "color":  "blue",
        },
    ]

    _hdr = st.columns([1, 5, 3])
    _hdr[0].markdown("**Hypothesis**")
    _hdr[1].markdown("**Description**")
    _hdr[2].markdown("**Status**")
    st.markdown("---")
    for _h in _hypotheses:
        _cols = st.columns([1, 5, 3])
        _cols[0].markdown(f"**{_h['id']}**")
        _cols[1].write(_h["desc"])
        _cols[2].markdown(
            f"<span style='color:{_h['color']};font-weight:bold'>{_h['status']}</span>",
            unsafe_allow_html=True
        )
        st.markdown("---")

    # ── Rest of About page (unchanged) ───────────────────────────────
    st.markdown("""
## Architecture

| Layer | Component | Technology |
|-------|-----------|------------|
| Data Ingestion | CSV loading from 3 datasets | pandas (prototype) |
| Feature Engineering | 11 quality metrics, 3 levels | Python / NumPy |
| ML Classification | Random Forest Classifier | scikit-learn |
| Visualisation | Interactive dashboard | Streamlit + Plotly |

---

## Datasets

| Dataset | Records | HIGH Quality | LOW Quality |
|---------|---------|-------------|------------|
| Titanic (Kaggle) | 89 | 76 | 13 |
| Brazilian E-Commerce (Kaggle) | 1,000 | 700 | 300 |
| HR Analytics (Kaggle) | 147 | 127 | 20 |
| **Total** | **1,236** | **903 (73%)** | **333 (27%)** |

---

## 11 Engineered Quality Features

**ROW-Level (3):** completeness_score, consistency_score, uniqueness_score

**COLUMN-Level (2):** validity_score, accuracy_score

**DATASET-Level (6):** conformity_score, timeliness_score, integrity_score,
schema_match_score, format_compliance_score, outlier_score

---

## References

1. Batini, C., & Scannapieco, M. (2016). *Data and Information Quality*. Springer.
2. Zhou, Y. et al. (2024). A survey on data quality dimensions for ML. *arXiv:2406.19614*.
3. Da Silva, L.M. et al. (2023). Data quality requirements for ML pipelines. *KDIR 2023*.
4. Ehrlinger, L. et al. (2023). ML-based analysis of heterogeneous data sources. *C&IE, 179*.
5. Esco, E. (2017). *Flexible Infrastructure Supporting ML for Anomaly Detection*. WPI.
""")
