import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def prepare_customer_data(customer_df, artifacts):
    """Prepare customer data for prediction"""
    df = customer_df.copy()
    
    # Get artifacts
    label_encoders = artifacts['label_encoders']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    
    # Handle missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in df.columns and col != 'churn':
            try:
                # Ensure values are strings and handle unknown categories
                df[col] = df[col].astype(str)
                # Transform using existing classes, set unknown to first class
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
            except Exception as e:
                print(f"Warning: Could not encode {col}: {e}")
                df[col] = 0
    
    # Scale numerical features
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges', 'age']
    for col in numerical_cols:
        if col in df.columns:
            try:
                df[col] = scaler.transform(df[[col]])
            except:
                # If scaling fails, use default
                df[col] = (df[col] - scaler.mean_[numerical_cols.index(col)]) / scaler.scale_[numerical_cols.index(col)]
    
    # Select only required features in correct order
    df = df[feature_names]
    
    return df

def predict_churn(customer_df, artifacts):
    """Predict churn for a single customer"""
    try:
        prepared_data = prepare_customer_data(customer_df, artifacts)
        model = artifacts['model']
        
        probability = float(model.predict_proba(prepared_data)[0][1])
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0, 0.0

def get_risk_factors(customer_df, probability, artifacts):
    """Identify key risk factors"""
    factors = {}
    
    if 'tenure' in customer_df.columns:
        tenure = customer_df['tenure'].iloc[0]
        if tenure < 12:
            factors['⚠️ Short tenure (<12 months)'] = min(1.0, (12 - tenure) / 12)
    
    if 'monthly_charges' in customer_df.columns:
        monthly_charges = customer_df['monthly_charges'].iloc[0]
        if monthly_charges > 70:
            factors['💰 High monthly charges'] = min(1.0, (monthly_charges - 70) / 50)
    
    if 'contract_type' in customer_df.columns:
        contract = customer_df['contract_type'].iloc[0]
        if contract == 'Month-to-month':
            factors['📄 Month-to-month contract'] = 0.8
    
    if 'payment_method' in customer_df.columns:
        payment = customer_df['payment_method'].iloc[0]
        if payment == 'Electronic check':
            factors['💳 Electronic check payment'] = 0.6
    
    if 'online_security' in customer_df.columns:
        security = customer_df['online_security'].iloc[0]
        if security == 'No':
            factors['🔒 No online security'] = 0.5
    
    # Add probability-based factor
    if probability > 0.7:
        factors['🎯 High risk score'] = probability
    
    return factors

def generate_recommendations(customer_df, probability):
    """Generate retention recommendations"""
    recommendations = []
    
    if probability > 0.7:
        recommendations.append("🚨 IMMEDIATE ACTION: Contact customer within 24 hours")
        recommendations.append("💰 Offer loyalty discount of 20-30% for 6 months")
        recommendations.append("⭐ Upgrade service tier at no additional cost")
        recommendations.append("📞 Schedule personal account review call")
        recommendations.append("🎁 Provide exclusive loyalty rewards")
    
    elif probability > 0.4:
        recommendations.append("📧 Send personalized retention email within 48 hours")
        recommendations.append("🎁 Offer 10-15% discount on next 3 months")
        recommendations.append("📱 Promote additional service bundles")
        recommendations.append("📊 Send customer satisfaction survey")
    
    else:
        recommendations.append("📊 Monitor usage patterns monthly")
        recommendations.append("📧 Send engagement content quarterly")
        recommendations.append("⭐ Encourage customer referrals with incentives")
        recommendations.append("🎯 Target for cross-selling opportunities")
    
    # Specific recommendations based on attributes
    if 'tenure' in customer_df.columns and customer_df['tenure'].iloc[0] < 6:
        recommendations.append("🆕 Implement enhanced new customer onboarding program")
        recommendations.append("📞 Weekly check-in calls for first month")
    
    if 'online_security' in customer_df.columns and customer_df['online_security'].iloc[0] == 'No':
        recommendations.append("🔒 Offer free online security trial for 6 months")
        recommendations.append("📚 Send security awareness materials")
    
    if 'tech_support' in customer_df.columns and customer_df['tech_support'].iloc[0] == 'No':
        recommendations.append("🛠️ Offer free tech support for 3 months trial")
    
    # Remove duplicates while preserving order
    unique_recs = []
    for rec in recommendations:
        if rec not in unique_recs:
            unique_recs.append(rec)
    
    return unique_recs[:8]  # Return top 8 recommendations