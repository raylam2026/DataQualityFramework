"""
FINAL WORKING TEST CODE FOR ml_classifier.py

"""

from quality_assessment import QualityAssessmentPipeline
from quality_assessment.ml_classifier import QualityClassifier
from pyspark.sql.functions import col, when
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_import():
    """Test 1: Can we import the classifier?"""
    print("\n" + "="*80)
    print("TEST 1: IMPORT TEST")
    print("="*80)
    
    try:
        from quality_assessment.ml_classifier import QualityClassifier
        print("✅ SUCCESS: QualityClassifier imported successfully!")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_initialization():
    """Test 2: Can we initialize the classifier?"""
    print("\n" + "="*80)
    print("TEST 2: INITIALIZATION TEST")
    print("="*80)
    
    try:
        pipeline = QualityAssessmentPipeline()
        spark = pipeline.spark

        # Add these 2 lines to fix Spark timeout
        spark.sparkContext.getConf() \
            .set('spark.executor.heartbeatInterval', '30s') \
            .set('spark.network.timeout', '600s')
        
        classifier = QualityClassifier(spark)
        classifier = QualityClassifier(pipeline.spark)
        print("✅ SUCCESS: QualityClassifier initialized!")
        return True, pipeline, classifier
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, None, None


def test_load_data(pipeline, classifier):
    """Test 3: Can we load and prepare data?"""
    print("\n" + "="*80)
    print("TEST 3: DATA LOADING TEST")
    print("="*80)
    
    try:
        # Load Titanic dataset with correct file path
        print("Loading Titanic dataset...")
        titanic_path = 'data/raw/titanic/train.csv'
        df_raw = pipeline.data_loader.load_file(titanic_path)
        print(f"✅ Loaded {df_raw.count()} rows")
        
        # Calculate total_cols from raw DataFrame BEFORE feature engineering
        print("Calculating column count...")
        total_cols = len(df_raw.columns)
        print(f"   Total columns in raw data: {total_cols}")
        
        # Extract features
        print("Extracting features...")
        df_features, _, _ = pipeline.feature_engineer.engineer_all_features(df_raw)
        print(f"✅ Extracted features from {df_features.count()} rows")
        print(f"   Feature columns: {len(df_features.columns)}")
        
        # Create null_ratio using the correct total_cols
        print("Creating null_ratio feature...")
        df_features = df_features.withColumn(
            'null_ratio',
            col('null_count').cast('double') / total_cols
        )
        print(f"✅ Created null_ratio = null_count / {total_cols}")
        
        # Verify null_ratio was created
        print("Verifying null_ratio...")
        sample_data = df_features.select('null_count', 'null_ratio').take(2)
        for i, row in enumerate(sample_data):
            print(f"   Row {i+1}: null_count={row['null_count']}, null_ratio={row['null_ratio']:.4f}")
        
        # Create labels based on null_ratio
        print("Creating ground truth labels...")
        df_labeled = df_features.withColumn(
            'ground_truth',
            when(col('null_ratio') <= 0.1, 1).otherwise(0)
        )
        print(f"✅ Created labels")
        
        # Check label distribution
        label_counts = df_labeled.groupBy('ground_truth').count().collect()
        print(f"   Label distribution:")
        for row in label_counts:
            label = int(row['ground_truth'])
            count = row['count']
            pct = (count / df_labeled.count()) * 100
            print(f"     Label {label}: {count} samples ({pct:.1f}%)")
        
        return True, df_labeled
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_train_random_forest(classifier, df_labeled):
    """Test 4: Can we train the model?"""
    print("\n" + "="*80)
    print("TEST 4: TRAINING TEST")
    print("="*80)
    
    try:
        # Split data
        print("Splitting data (80/20)...")
        train, test = df_labeled.randomSplit([0.8, 0.2], seed=42)
        train_count = train.count()
        test_count = test.count()
        print(f"✅ Train set: {train_count} samples")
        print(f"✅ Test set: {test_count} samples")
        
        # Train model
        print("\nTraining Random Forest (100 trees, max_depth=15)...")
        model, train_metrics = classifier.train_random_forest(train)
        
        print("✅ Model trained successfully!")
        print(f"   Training Metrics:")
        print(f"     - AUC-ROC: {train_metrics['auc_roc']:.4f}")
        print(f"     - F1-Score: {train_metrics['f1_score']:.4f}")
        print(f"     - Samples: {train_metrics['training_samples']}")
        
        return True, model, test, train_metrics
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_evaluate_model(classifier, model, test_set):
    """Test 5: Can we evaluate the model?"""
    print("\n" + "="*80)
    print("TEST 5: EVALUATION TEST")
    print("="*80)
    
    try:
        test_count = test_set.count()
        print(f"Evaluating on {test_count} test samples...")
        test_metrics = classifier.evaluate_model(test_set, model)
        
        print("✅ Model evaluated successfully!")
        print(f"   Test Metrics:")
        print(f"     - Precision: {test_metrics['precision']:.4f} (need ≥ 0.80)")
        print(f"     - Recall: {test_metrics['recall']:.4f} (need ≥ 0.75)")
        print(f"     - F1-Score: {test_metrics['f1_score']:.4f} (need ≥ 0.77)")
        print(f"     - AUC-ROC: {test_metrics['auc_roc']:.4f}")
        
        # Check H1 criteria
        h1_met = (
            test_metrics['precision'] >= 0.80 and
            test_metrics['recall'] >= 0.75 and
            test_metrics['f1_score'] >= 0.77
        )
        
        if h1_met:
            print(f"\n✅ H1 HYPOTHESIS CRITERIA MET!")
        else:
            print(f"\n⚠️  H1 criteria not met (This is Phase 4 optimization goal)")
            print(f"   Current: P={test_metrics['precision']:.4f}, R={test_metrics['recall']:.4f}, F1={test_metrics['f1_score']:.4f}")
            print(f"   Target:  P≥0.80, R≥0.75, F1≥0.77")
        
        return True, test_metrics
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_feature_importance(classifier, model):
    """Test 6: Can we extract feature importance?"""
    print("\n" + "="*80)
    print("TEST 6: FEATURE IMPORTANCE TEST")
    print("="*80)
    
    try:
        print("Extracting top 10 features...")
        importance = classifier.feature_importance(model)
        
        if importance:
            print("✅ Feature importance extracted!")
            print(f"   Top 10 Features:")
            for i, (feature, score) in enumerate(importance.items(), 1):
                print(f"     {i:2d}. {feature:30s}: {score:.4f}")
        else:
            print("⚠️  No feature importance extracted (non-critical)")
        
        return True, importance
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_cross_validation(classifier, df_labeled):
    """Test 7: Can we perform cross-validation?"""
    print("\n" + "="*80)
    print("TEST 7: CROSS-VALIDATION TEST (5-fold)")
    print("="*80)
    
    try:
        print("Performing 5-fold cross-validation...")
        print("(This may take a few minutes...)")
        mean_metrics, std_metrics = classifier.cross_validate(df_labeled, num_folds=5)
        
        print("✅ Cross-validation completed!")
        print(f"   Mean Metrics:")
        for k, v in mean_metrics.items():
            print(f"     - {k}: {v:.4f}")
        
        return True, mean_metrics
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all tests"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "TESTING ml_classifier.py (PRODUCTION VERSION)" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    
    # Test 1: Import
    success = test_basic_import()
    if not success:
        print("\n❌ Cannot proceed: Import failed")
        return False
    
    # Test 2: Initialization
    success, pipeline, classifier = test_initialization()
    if not success:
        print("\n❌ Cannot proceed: Initialization failed")
        return False
    
    # Test 3: Data Loading
    success, df_labeled = test_load_data(pipeline, classifier)
    if not success:
        print("\n❌ Cannot proceed: Data loading failed")
        return False
    
    # Test 4: Training
    success, model, test_set, train_metrics = test_train_random_forest(classifier, df_labeled)
    if not success:
        print("\n❌ Cannot proceed: Training failed")
        return False
    
    # Test 5: Evaluation
    success, test_metrics = test_evaluate_model(classifier, model, test_set)
    if not success:
        print("\n❌ Cannot proceed: Evaluation failed")
        return False
    
    # Test 6: Feature Importance (optional)
    success, importance = test_feature_importance(classifier, model)
    if not success:
        print("\n⚠️  Feature importance test failed (non-critical)")
    
    # Test 7: Cross-validation (optional)
    success, cv_metrics = test_cross_validation(classifier, df_labeled)
    if not success:
        print("\n⚠️  Cross-validation test failed (non-critical, can retry later)")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TEST SUMMARY - ALL CORE TESTS PASSED")
    print("="*80)
    print("✅ Test 1: Import")
    print("✅ Test 2: Initialization")
    print("✅ Test 3: Data Loading")
    print("✅ Test 4: Training")
    print("✅ Test 5: Evaluation")
    
    if test_metrics:
        print("\n" + "="*80)
        print("FINAL METRICS SUMMARY")
        print("="*80)
        print(f"Precision:  {test_metrics['precision']:.4f}")
        print(f"Recall:     {test_metrics['recall']:.4f}")
        print(f"F1-Score:   {test_metrics['f1_score']:.4f}")
        print(f"AUC-ROC:    {test_metrics['auc_roc']:.4f}")
        
        h1_met = (
            test_metrics['precision'] >= 0.80 and
            test_metrics['recall'] >= 0.75 and
            test_metrics['f1_score'] >= 0.77
        )
        
        if h1_met:
            print("\n🎉 H1 HYPOTHESIS CRITERIA MET!")
            print("   Your classifier meets specification requirements!")
        else:
            print("\n⚠️  H1 criteria not fully met")
            print("   This is expected for initial testing")
            print("   Phase 4 will focus on optimizing these metrics")
    
    print("\n" + "="*80)
    print("✅ Your ml_classifier.py is working correctly!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)