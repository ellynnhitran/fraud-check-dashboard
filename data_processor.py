import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'merchant', 'category', 'amt', 'gender', 'city', 'state'
        ]
        self.all_columns = [
            'merchant', 'category', 'amt', 'cc_num', 'first', 'last', 'gender',
            'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job',
            'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'
        ]
        
    def validate_transaction_data(self, data):
        """Validate transaction data format and completeness"""
        errors = []
        
        # Convert to dict if it's a pandas Series
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        # Check only the essential required fields
        missing_fields = []
        for field in self.required_columns:
            if field not in data or data[field] is None or str(data[field]).strip() == '':
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate data types and ranges
        try:
            # Amount validation
            if 'amt' in data:
                amount = float(data['amt'])
                if amount <= 0:
                    errors.append("Amount must be positive")
                if amount > 50000:
                    errors.append("Amount seems unusually high (>$50,000)")
            
            # Coordinate validation
            if 'lat' in data:
                lat = float(data['lat'])
                if not (-90 <= lat <= 90):
                    errors.append("Latitude must be between -90 and 90")
            
            if 'long' in data:
                long_val = float(data['long'])
                if not (-180 <= long_val <= 180):
                    errors.append("Longitude must be between -180 and 180")
            
            # City population validation
            if 'city_pop' in data:
                city_pop = int(data['city_pop'])
                if city_pop <= 0:
                    errors.append("City population must be positive")
            
            # Gender validation
            if 'gender' in data:
                if str(data['gender']).upper() not in ['M', 'F', 'MALE', 'FEMALE']:
                    errors.append("Gender must be M, F, Male, or Female")
            
            # Date of birth validation
            if 'dob' in data:
                try:
                    if isinstance(data['dob'], str):
                        dob = datetime.strptime(data['dob'], '%Y-%m-%d')
                    else:
                        dob = data['dob']
                    
                    if dob > datetime.now():
                        errors.append("Date of birth cannot be in the future")
                    
                    age = (datetime.now() - dob).days / 365.25
                    if age < 18 or age > 120:
                        errors.append("Age must be between 18 and 120 years")
                        
                except (ValueError, TypeError):
                    errors.append("Invalid date of birth format (use YYYY-MM-DD)")
            
        except (ValueError, TypeError) as e:
            errors.append(f"Data type validation error: {str(e)}")
        
        return errors
    
    def clean_and_standardize(self, data):
        """Clean and standardize transaction data"""
        # Convert to dict if it's a pandas Series
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        cleaned_data = data.copy()
        
        # Auto-generate missing required fields with reasonable defaults
        if 'first' not in cleaned_data or not cleaned_data['first']:
            cleaned_data['first'] = 'Customer'
        if 'last' not in cleaned_data or not cleaned_data['last']:
            cleaned_data['last'] = 'User'
        if 'street' not in cleaned_data or not cleaned_data['street']:
            cleaned_data['street'] = '123 Main St'
        if 'zip' not in cleaned_data or not cleaned_data['zip']:
            cleaned_data['zip'] = '10001'
        if 'lat' not in cleaned_data or not cleaned_data['lat']:
            cleaned_data['lat'] = 40.7128
        if 'long' not in cleaned_data or not cleaned_data['long']:
            cleaned_data['long'] = -74.0060
        if 'city_pop' not in cleaned_data or not cleaned_data['city_pop']:
            cleaned_data['city_pop'] = 100000
        if 'job' not in cleaned_data or not cleaned_data['job']:
            cleaned_data['job'] = 'Professional'
        if 'dob' not in cleaned_data or not cleaned_data['dob']:
            cleaned_data['dob'] = '1980-01-01'
        if 'merch_lat' not in cleaned_data or not cleaned_data['merch_lat']:
            cleaned_data['merch_lat'] = 40.7128
        if 'merch_long' not in cleaned_data or not cleaned_data['merch_long']:
            cleaned_data['merch_long'] = -74.0060
        
        # Standardize text fields
        text_fields = ['merchant', 'first', 'last', 'street', 'city', 'state', 'job']
        for field in text_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = str(cleaned_data[field]).strip().title()
        
        # Standardize gender
        if 'gender' in cleaned_data:
            gender = str(cleaned_data['gender']).upper()
            if gender in ['MALE', 'M']:
                cleaned_data['gender'] = 'M'
            elif gender in ['FEMALE', 'F']:
                cleaned_data['gender'] = 'F'
        
        # Standardize state codes
        if 'state' in cleaned_data:
            cleaned_data['state'] = str(cleaned_data['state']).upper()[:2]
        
        # Clean zip code
        if 'zip' in cleaned_data:
            zip_code = str(cleaned_data['zip'])
            # Extract just the 5-digit ZIP code
            zip_match = re.search(r'\d{5}', zip_code)
            if zip_match:
                cleaned_data['zip'] = zip_match.group()
            else:
                cleaned_data['zip'] = '00000'  # Default for invalid ZIP
        
        # Ensure numeric fields are properly typed
        numeric_fields = {
            'amt': float,
            'cc_num': int,
            'lat': float,
            'long': float,
            'city_pop': int,
            'unix_time': int,
            'merch_lat': float,
            'merch_long': float
        }
        
        for field, dtype in numeric_fields.items():
            if field in cleaned_data:
                try:
                    cleaned_data[field] = dtype(cleaned_data[field])
                except (ValueError, TypeError):
                    # Set default values for invalid numeric data
                    if dtype == float:
                        cleaned_data[field] = 0.0
                    else:
                        cleaned_data[field] = 0
        
        # Handle date of birth
        if 'dob' in cleaned_data:
            try:
                if isinstance(cleaned_data['dob'], str):
                    cleaned_data['dob'] = datetime.strptime(cleaned_data['dob'], '%Y-%m-%d').strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                # Default to a reasonable date if parsing fails
                cleaned_data['dob'] = '1980-01-01'
        
        # Generate missing transaction identifiers if needed
        if 'trans_num' not in cleaned_data or not cleaned_data['trans_num']:
            cleaned_data['trans_num'] = f"T{np.random.randint(100000, 999999)}"
        
        if 'cc_num' not in cleaned_data or not cleaned_data['cc_num']:
            cleaned_data['cc_num'] = np.random.randint(1000000000000000, 9999999999999999)
        
        if 'unix_time' not in cleaned_data or not cleaned_data['unix_time']:
            cleaned_data['unix_time'] = int(datetime.now().timestamp())
        
        return cleaned_data
    
    def process_single_transaction(self, data):
        """Process a single transaction for prediction"""
        # Validate data
        validation_errors = self.validate_transaction_data(data)
        if validation_errors:
            raise ValueError(f"Data validation failed: {'; '.join(validation_errors)}")
        
        # Clean and standardize
        cleaned_data = self.clean_and_standardize(data)
        
        return cleaned_data
    
    def process_batch_csv(self, csv_file):
        """Process a batch CSV file"""
        try:
            # Read CSV file
            if hasattr(csv_file, 'read'):
                df = pd.read_csv(csv_file)
            else:
                df = pd.read_csv(csv_file)
            
            # Validate only essential columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Process each row
            processed_data = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    processed_row = self.process_single_transaction(row.to_dict())
                    processed_data.append(processed_row)
                except Exception as e:
                    errors.append(f"Row {idx + 1}: {str(e)}")
            
            if errors:
                raise ValueError(f"Processing errors: {'; '.join(errors[:5])}")  # Show first 5 errors
            
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            raise ValueError(f"CSV processing error: {str(e)}")
    
    def get_column_mapping(self):
        """Get mapping of expected columns and their descriptions"""
        column_descriptions = {
            'merchant': 'Merchant name (text)',
            'category': 'Transaction category (e.g., grocery_pos, gas_transport)',
            'amt': 'Transaction amount (decimal)',
            'gender': 'Gender (M/F)',
            'city': 'City name (text)',
            'state': 'State code (2 letters)'
        }
        
        return column_descriptions
