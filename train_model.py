import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
def create_directories():
    """Create required directories if they don't exist"""
    directories = ['models', 'data']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")

def create_sample_data():
    """Create sample customer data if real data not available"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn based on logical patterns
    churn_probability = (
        (df['contract_type'] == 'Month-to-month') * 0.3 +
        (df['tenure'] < 12) * 0.25 +
        (df['monthly_charges'] > 70) * 0.2 +
        (df['payment_method'] == 'Electronic check') * 0.15 +
        (df['online_security'] == 'No') * 0.1
    )
    
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    df_processed = df.copy()
    
    # Drop customer_id
    if 'customer_id' in df_processed.columns:
        df_processed = df_processed.drop('customer_id', axis=1)
    
    # Handle categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_cols:
        if col != 'churn':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Separate features and target
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, label_encoders, scaler

def train_models(X, y):
    """Train and compare multiple models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_balanced, y_train_balanced)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    lr_results = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'auc_roc': roc_auc_score(y_test, lr_proba)
    }
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_results = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'auc_roc': roc_auc_score(y_test, xgb_proba),
        'model': xgb_model
    }
    
    # Print results
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Logistic Regression - Accuracy: {lr_results['accuracy']:.3f}, Recall: {lr_results['recall']:.3f}, AUC-ROC: {lr_results['auc_roc']:.3f}")
    print(f"XGBoost           - Accuracy: {xgb_results['accuracy']:.3f}, Recall: {xgb_results['recall']:.3f}, AUC-ROC: {xgb_results['auc_roc']:.3f}")
    
    # Print detailed classification report for XGBoost
    print("\n" + "="*50)
    print("XGBOOST CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, xgb_pred, target_names=['Stayed', 'Churned']))
    
    return xgb_results['model'], X_test, y_test, X.columns.tolist()

def save_model_and_artifacts(model, feature_names, label_encoders, scaler):
    """Save model and preprocessing artifacts"""
    artifacts = {
        'model': model,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    model_path = 'models/churn_model.pkl'
    joblib.dump(artifacts, model_path)
    print(f"\n✅ Model and artifacts saved to '{model_path}'")
    
    # Verify save was successful
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024  # Size in KB
        print(f"   File size: {file_size:.2f} KB")
    else:
        print("❌ Error: Model file was not created")

def main():
    print("🚀 Starting Churn Prediction Pipeline")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Load or create data
    data_path = 'data/customer_data.csv'
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✅ Loaded {len(df)} customer records from {data_path}")
        else:
            raise FileNotFoundError(f"Data file not found at {data_path}")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"⚠️ {e}")
        print("📊 Creating sample data...")
        df = create_sample_data()
        df.to_csv(data_path, index=False)
        print(f"✅ Created {len(df)} sample customer records saved to {data_path}")
    
    # Display data info
    print("\n📊 Dataset Overview:")
    print(f"   - Total customers: {len(df)}")
    print(f"   - Features: {len(df.columns) - 1}")
    print(f"   - Churn rate: {df['churn'].mean()*100:.1f}%")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Preprocess
    print("\n🔄 Preprocessing data...")
    X, y, label_encoders, scaler = preprocess_data(df)
    print(f"   ✅ Preprocessing complete")
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target distribution: {y.value_counts().to_dict()}")
    
    # Train models
    print("\n🤖 Training models...")
    best_model, X_test, y_test, feature_names = train_models(X, y)
    
    # Save artifacts
    save_model_and_artifacts(best_model, feature_names, label_encoders, scaler)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to launch the dashboard")
    print("2. Upload customer data for predictions")
    print("3. View churn analytics and recommendations")
    
    return best_model

if __name__ == "__main__":
    main()