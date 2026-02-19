# Adaptive ML Framework for Data Quality Assessment

**Student:** LAM CHIT WUI (27806)  
**University:** University of Liverpool  
**Program:** MSc Data Science & AI  
**Advisor:** Dr. Petros Lalos  
**Instructor:** Dr. Andrea Corradini  

---

## 🎯 Phase 6: Production Pipeline - COMPLETE ✅

### Overview

A PySpark-based machine learning framework for **automated data quality assessment**
in heterogeneous big data repositories. Implements Random Forest classification with
5-fold cross-validation for detecting and categorising quality issues across structured,
semi-structured, and categorical datasets.

### Architecture (5-Layer)

1. **Data Ingestion** — PySpark multi-format loader (CSV/JSON) with schema inference
2. **Feature Computation** — 11 quality features at ROW, COLUMN, and DATASET levels
3. **ML Classification** — Random Forest (100 trees, max_depth=15, 70/30 split)
4. **Evaluation** — Precision, Recall, F1-score, ROC-AUC, 5-fold cross-validation
5. **Dashboard** — Streamlit app with quality scoring, predictions, and file upload

### Datasets

| Dataset | Records | Columns | Type |
|---------|---------|---------|------|
| Titanic | 891 | 12 | Structured CSV |
| Brazilian E-Commerce | 99K+ | Multi-table | Semi-structured |
| HR Analytics | 1,470 | 35 | Categorical-heavy |

### Quick Start

```bash
pip install -r requirements.txt
cd phase6_production/pipeline
python ml_classifier.py          # Train the model
streamlit run streamlit_dashboard.py  # Launch dashboard
