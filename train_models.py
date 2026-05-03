"""
Model Training and Persistence Module
Trains and saves the best performing models for use in the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

def load_and_prepare_data():
    """Load dataset and prepare for training"""
    print("Loading dataset...")
    df = pd.read_csv('Credit_Card_Default.csv')
    
    # Get target column
    target_col = [col for col in df.columns if 'default' in col.lower()][0]
    
    # Prepare features and target
    X = df.drop(columns=[target_col, 'ID'], errors='ignore')
    y = df[target_col]
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_smote, X_test_scaled, y_train_smote, y_test, scaler, X_train.columns, target_col

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    models = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"  Test Accuracy: {rf_score:.4f}")
    models['random_forest'] = {
        'model': rf,
        'accuracy': rf_score,
        'type': 'ensemble'
    }
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    print(f"  Test Accuracy: {xgb_score:.4f}")
    models['xgboost'] = {
        'model': xgb_model,
        'accuracy': xgb_score,
        'type': 'boosting'
    }
    
    # LightGBM
    print("Training LightGBM...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=42,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
    lgbm.fit(X_train, y_train)
    lgbm_score = lgbm.score(X_test, y_test)
    print(f"  Test Accuracy: {lgbm_score:.4f}")
    models['lightgbm'] = {
        'model': lgbm,
        'accuracy': lgbm_score,
        'type': 'boosting'
    }
    
    return models

def save_models(models, scaler, feature_names):
    """Save trained models and preprocessing objects"""
    print("\n" + "="*80)
    print("SAVING MODELS AND ARTIFACTS")
    print("="*80)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Find best model
    best_model_name = max(models.keys(), key=lambda x: models[x]['accuracy'])
    best_model = models[best_model_name]['model']
    
    # Save models
    for model_name, model_info in models.items():
        filepath = f"models/{model_name}_model.pkl"
        joblib.dump(model_info['model'], filepath)
        print(f"✓ Saved {model_name}: {filepath}")
    
    # Save scaler
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save feature names
    features_path = "models/feature_names.pkl"
    joblib.dump(feature_names.tolist(), features_path)
    print(f"✓ Saved feature names: {features_path}")
    
    # Save best model info
    best_model_path = "models/best_model.pkl"
    best_model_info = {
        'model': best_model,
        'name': best_model_name,
        'accuracy': models[best_model_name]['accuracy']
    }
    joblib.dump(best_model_info, best_model_path)
    print(f"✓ Saved best model: {best_model_path}")
    print(f"\n  Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"  Accuracy: {models[best_model_name]['accuracy']:.4f}")
    
    return best_model_name

def load_models():
    """Load saved models and artifacts"""
    print("Loading saved models...")
    
    models = {}
    
    if os.path.exists("models/best_model.pkl"):
        best_model_info = joblib.load("models/best_model.pkl")
        models['best'] = best_model_info
        print(f"✓ Loaded best model: {best_model_info['name']}")
    
    if os.path.exists("models/scaler.pkl"):
        models['scaler'] = joblib.load("models/scaler.pkl")
        print("✓ Loaded scaler")
    
    if os.path.exists("models/feature_names.pkl"):
        models['features'] = joblib.load("models/feature_names.pkl")
        print(f"✓ Loaded {len(models['features'])} feature names")
    
    return models

def predict_default(customer_data, models):
    """Make prediction for a customer"""
    
    if 'best' not in models or 'scaler' not in models or 'features' not in models:
        raise ValueError("Models not properly loaded")
    
    # Prepare data
    scaler = models['scaler']
    feature_names = models['features']
    model = models['best']['model']
    
    # Create DataFrame from customer data
    customer_df = pd.DataFrame([customer_data])
    
    # Ensure proper column order and fill missing columns with 0
    for col in feature_names:
        if col not in customer_df.columns:
            customer_df[col] = 0
    
    customer_df = customer_df[feature_names]
    
    # Scale features
    customer_scaled = scaler.transform(customer_df)
    
    # Make prediction
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    }

if __name__ == "__main__":
    print("="*80)
    print("CREDIT CARD DEFAULT PREDICTION - MODEL TRAINING")
    print("="*80)
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, scaler, feature_names, target_col = load_and_prepare_data()
        print(f"\n✓ Data prepared:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Testing set: {X_test.shape}")
        print(f"  Features: {len(feature_names)}")
        
        # Train models
        models = train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        best_model_name = save_models(models, scaler, feature_names)
        
        print("\n" + "="*80)
        print("✓ MODEL TRAINING COMPLETE")
        print("="*80)
        print("\nModels are ready for use in the Streamlit app!")
        print("Run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()
