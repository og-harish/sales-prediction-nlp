import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def train_and_save_model(cleaned_df, model_dir='models'):
    """
    Step 4: Machine Learning Module. Trains and compares 3 models.
    Saves the best model and preprocessors to models/sales_model.pkl.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Define Features & Target
    features = [
        'Quantity', 'Unit_Price_INR', 'discount_pct', 
        'State', 'City', 'Product', 
        'year', 'month', 'day', 'weekend'
    ]
    target = 'Sales_Value_INR'
    
    # Ensure all columns exist in df
    for col in features + [target]:
        if col not in cleaned_df.columns:
            raise ValueError(f"Required column '{col}' is missing from the cleaned dataframe.")
            
    X = cleaned_df[features].copy()
    y = cleaned_df[target].copy()
    
    # 2. Encode Categorical Features
    categorical_features = ['State', 'City', 'Product']
    encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # 4. Initialize Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    performance = {}
    trained_models = {}
    
    # 5. Train and Evaluate
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        performance[name] = {
            'R2': float(r2),
            'MAE': float(mae),
            'RMSE': float(rmse)
        }
        
    # 6. Select the Best Model automatically based on highest R2 Score
    best_model_name = max(performance, key=lambda k: performance[k]['R2'])
    best_model = trained_models[best_model_name]
    
    print("--- Model Performance Comparison ---")
    for name, metrics in performance.items():
        print(f"{name}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}")
    print(f"Selected Best Model: {best_model_name}")
    
    # 7. Save Best Model and Metadata
    model_state = {
        'model': best_model,
        'model_name': best_model_name,
        'encoders': encoders,
        'features': features,
        'categorical_features': categorical_features,
        'performance': performance,
        'best_performance': performance[best_model_name]
    }
    
    model_path = os.path.join(model_dir, 'sales_model.pkl')
    joblib.dump(model_state, model_path)
    print(f"Saved model and preprocessors successfully to {model_path}")
    
    return model_state

def predict_sales(input_data, model_path='models/sales_model.pkl'):
    """
    Step 5: Load the saved model state and predict sales for a single input record.
    input_data should be a dictionary containing:
    {
      'State': str, 'City': str, 'Product': str,
      'Quantity': float, 'Unit_Price_INR': float, 'discount_pct': float,
      'Date': pd.Timestamp / datetime
    }
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
        
    # Load model state
    model_state = joblib.load(model_path)
    model = model_state['model']
    encoders = model_state['encoders']
    features = model_state['features']
    
    # Extract date features
    dt = pd.to_datetime(input_data['Date'])
    
    # Construct raw feature row
    raw_row = {
        'Quantity': float(input_data['Quantity']),
        'Unit_Price_INR': float(input_data['Unit_Price_INR']),
        'discount_pct': float(input_data['discount_pct']),
        'State': str(input_data['State']).strip(),
        'City': str(input_data['City']).strip(),
        'Product': str(input_data['Product']).strip().lower(),
        'year': int(dt.year),
        'month': int(dt.month),
        'day': int(dt.day),
        'weekend': int(1 if dt.weekday() in [5, 6] else 0)
    }
    
    # Encode categorical features
    encoded_row = {}
    for col in features:
        val = raw_row[col]
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels by mapping them to closest match or default class
            try:
                # Value matches exactly
                encoded_row[col] = le.transform([str(val)])[0]
            except ValueError:
                # If label not found during fit, fall back to first label or default index 0
                encoded_row[col] = 0
        else:
            encoded_row[col] = val
            
    # Convert to DataFrame in exact feature order
    df_row = pd.DataFrame([encoded_row])[features]
    
    # Run prediction
    prediction = model.predict(df_row)[0]
    
    # Ensure prediction is non-negative
    prediction = max(0.0, float(prediction))
    
    # Return predicted value and model metrics for confidence calculation
    return prediction, model_state
