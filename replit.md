# Credit Card Fraud Detection System

## Overview

This is a comprehensive credit card fraud detection system built with Streamlit that provides real-time fraud analysis capabilities. The application combines machine learning-based fraud detection with an intuitive web interface, allowing users to analyze transactions through manual input or bulk CSV processing. The system uses XGBoost for classification and includes synthetic data generation for model training when historical data is unavailable.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with a responsive layout using Streamlit's column system and sidebar navigation
- **Interactive Components**: Manual transaction input forms, CSV upload functionality, and real-time risk indicators with color-coded alerts
- **Session State Management**: Persistent storage of fraud detector instances, prediction history, and data processor objects across user sessions

### Backend Architecture
- **Modular Design**: Separated concerns across multiple Python modules (fraud_detector.py, model_trainer.py, data_processor.py, csv_template.py)
- **Model Management**: Automatic model loading/training pipeline with pickle serialization for model persistence
- **Data Processing Pipeline**: Comprehensive validation and preprocessing system for both single transactions and batch processing

### Machine Learning Components
- **XGBoost Classifier**: Primary fraud detection algorithm chosen for its effectiveness with tabular data and imbalanced datasets
- **Feature Engineering**: Automated encoding of categorical variables and standardization of numerical features
- **SMOTE Integration**: Synthetic Minority Oversampling Technique to handle class imbalance in fraud detection
- **Synthetic Data Generation**: Built-in capability to generate realistic transaction data for training when real data is unavailable

### Data Storage Solutions
- **Model Persistence**: Local file storage using pickle for trained models and preprocessing components
- **In-Memory Processing**: Pandas DataFrames for data manipulation and CSV processing
- **Session State**: Streamlit's session state for maintaining user context and prediction history

### Validation and Security
- **Data Validation**: Comprehensive input validation including range checks, data type validation, and required field verification
- **Error Handling**: Graceful error handling throughout the pipeline with user-friendly error messages
- **Risk Assessment**: Multi-level risk classification (High/Medium/Low/Safe) with probability thresholds

## External Dependencies

### Machine Learning Libraries
- **XGBoost**: Primary classification algorithm for fraud detection
- **scikit-learn**: Feature preprocessing, model evaluation metrics, and train-test splitting
- **imbalanced-learn (imblearn)**: SMOTE implementation for handling class imbalance

### Data Processing
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical computations and array operations

### Web Framework
- **Streamlit**: Complete web application framework providing UI components, session management, and deployment capabilities

### Utility Libraries
- **Pickle**: Model serialization and persistence
- **IO**: In-memory file operations for CSV processing
- **Datetime**: Timestamp handling and date operations
- **OS**: File system operations for model storage
- **Re**: Regular expression operations for data validation

### Development Tools
- **Python Standard Library**: Core functionality including file operations, data structures, and error handling

Note: The system is designed to be self-contained with synthetic data generation capabilities, making it deployable without external databases or APIs. All model training and prediction occurs locally within the application environment.