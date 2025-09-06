import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic credit card transaction data for training"""
        np.random.seed(42)
        
        # Categories and merchants
        categories = [
            'grocery_pos', 'gas_transport', 'misc_net', 'grocery_net', 'shopping_net',
            'shopping_pos', 'entertainment', 'food_dining', 'personal_care', 'health_fitness',
            'travel', 'kids_pets', 'home', 'misc_pos'
        ]
        
        merchants = [
            'Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonalds', 'Shell', 'Exxon',
            'Home Depot', 'CVS', 'Walgreens', 'Best Buy', 'Costco', 'Safeway', 'Kroger'
        ]
        
        jobs = [
            'Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Manager', 'Sales', 'Student',
            'Retired', 'Unemployed', 'Artist', 'Writer', 'Nurse', 'Police', 'Firefighter'
        ]
        
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        
        data = []
        
        for i in range(n_samples):
            # Basic transaction info
            merchant = np.random.choice(merchants)
            category = np.random.choice(categories)
            
            # Amount - different patterns for fraud vs legitimate
            is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate
            
            if is_fraud:
                # Fraud transactions tend to be higher amounts or very small amounts
                if np.random.random() < 0.7:
                    amount = np.random.exponential(500) + 200  # Higher amounts
                else:
                    amount = np.random.uniform(1, 50)  # Very small amounts
            else:
                # Legitimate transactions follow normal patterns
                if category in ['grocery_pos', 'grocery_net']:
                    amount = np.random.gamma(2, 50)
                elif category in ['gas_transport']:
                    amount = np.random.gamma(2, 30)
                elif category in ['entertainment', 'food_dining']:
                    amount = np.random.gamma(1.5, 40)
                else:
                    amount = np.random.gamma(2, 60)
            
            amount = max(1.0, min(amount, 5000.0))  # Cap at reasonable values
            
            # Personal info
            first_name = f"Person{i}"
            last_name = f"LastName{i}"
            gender = np.random.choice(['M', 'F'])
            
            # Location
            state = np.random.choice(states)
            city = f"City{np.random.randint(1, 100)}"
            zip_code = np.random.randint(10000, 99999)
            street = f"{np.random.randint(1, 9999)} {np.random.choice(['Main', 'First', 'Second', 'Oak', 'Pine'])} St"
            
            # Coordinates - US boundaries approximately
            lat = np.random.uniform(25.0, 48.0)
            long = np.random.uniform(-125.0, -65.0)
            
            # Merchant coordinates - sometimes far from customer (fraud indicator)
            if is_fraud and np.random.random() < 0.6:
                # Fraud: merchant far from customer
                merch_lat = np.random.uniform(25.0, 48.0)
                merch_long = np.random.uniform(-125.0, -65.0)
            else:
                # Legitimate: merchant near customer
                merch_lat = lat + np.random.normal(0, 0.5)
                merch_long = long + np.random.normal(0, 0.5)
            
            city_pop = np.random.randint(10000, 1000000)
            job = np.random.choice(jobs)
            
            # Date of birth
            birth_year = np.random.randint(1940, 2000)
            dob = f"{birth_year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            
            # Transaction identifiers
            cc_num = np.random.randint(1000000000000000, 9999999999999999)
            trans_num = f"T{i:06d}"
            
            # Unix time - random time in last year
            base_time = datetime.now().timestamp()
            unix_time = int(base_time - np.random.randint(0, 365 * 24 * 3600))
            
            transaction = {
                'merchant': merchant,
                'category': category,
                'amt': round(amount, 2),
                'cc_num': cc_num,
                'first': first_name,
                'last': last_name,
                'gender': gender,
                'street': street,
                'city': city,
                'state': state,
                'zip': zip_code,
                'lat': lat,
                'long': long,
                'city_pop': city_pop,
                'job': job,
                'dob': dob,
                'trans_num': trans_num,
                'unix_time': unix_time,
                'merch_lat': merch_lat,
                'merch_long': merch_long,
                'is_fraud': is_fraud
            }
            
            data.append(transaction)
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """Engineer features from raw transaction data"""
        df_processed = df.copy()
        
        # Time-based features
        df_processed['transaction_time'] = pd.to_datetime(df_processed['unix_time'], unit='s')
        df_processed['hour_of_day'] = df_processed['transaction_time'].dt.hour
        df_processed['day_of_week'] = df_processed['transaction_time'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        
        # Age calculation
        df_processed['dob'] = pd.to_datetime(df_processed['dob'])
        df_processed['age'] = (df_processed['transaction_time'] - df_processed['dob']).dt.days / 365.25
        
        # Distance between customer and merchant
        df_processed['distance'] = np.sqrt(
            (df_processed['lat'] - df_processed['merch_lat'])**2 + 
            (df_processed['long'] - df_processed['merch_long'])**2
        )
        
        # Amount-based features
        df_processed['log_amount'] = np.log1p(df_processed['amt'])
        df_processed['amount_per_pop'] = df_processed['amt'] / df_processed['city_pop']
        
        # Categorical encoding
        categorical_columns = ['merchant', 'category', 'gender', 'state', 'job']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                # Handle unseen categories
                df_processed[f'{col}_encoded'] = df_processed[col].astype(str).map(
                    dict(zip(self.label_encoders[col].classes_, self.label_encoders[col].transform(self.label_encoders[col].classes_)))
                ).fillna(-1)
        
        # Select features for training
        feature_columns = [
            'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'hour_of_day', 'day_of_week', 'is_weekend', 'age', 'distance',
            'log_amount', 'amount_per_pop'
        ] + [f'{col}_encoded' for col in categorical_columns]
        
        self.feature_columns = feature_columns
        
        return df_processed[feature_columns + ['is_fraud'] if 'is_fraud' in df_processed.columns else feature_columns]
    
    def train_model(self, data=None):
        """Train the XGBoost model"""
        if data is None:
            # Generate synthetic data for training
            data = self.generate_synthetic_data(50000)
        
        # Engineer features
        processed_data = self.engineer_features(data)
        
        # Prepare features and target
        X = processed_data[self.feature_columns]
        y = processed_data['is_fraud'] if 'is_fraud' in processed_data.columns else None
        
        if y is None:
            raise ValueError("No target variable 'is_fraud' found in data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train_balanced)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'test_accuracy': (y_pred == y_test).mean(),
            'test_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def save_model(self, filepath='fraud_detection_model.pkl'):
        """Save trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fraud_detection_model.pkl'):
        """Load trained model and preprocessors"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            
            print(f"Model loaded from {filepath}")
            return True
        return False
