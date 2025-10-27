# 🎵 Spotify Churn Analysis - 2025

A comprehensive machine learning pipeline for predicting user churn on Spotify using exploratory data analysis, feature engineering, and advanced modeling techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting user churn on Spotify. It includes:

- **Automated data exploration** with comprehensive visualizations
- **Feature engineering** and data cleaning
- **Multiple baseline models** (Logistic Regression, Random Forest, Gradient Boosting)
- **Hyperparameter tuning** with grid search
- **Model evaluation** with detailed metrics and visualizations

## ✨ Features

### 1. Exploratory Data Analysis (EDA)
- Churn distribution analysis
- Feature correlation analysis
- Platform and subscription insights
- User engagement metrics visualization

### 2. Data Cleaning & Feature Engineering
- Missing value imputation
- Outlier detection and capping using IQR method
- Feature creation:
  - `engagement_score`: Combined user engagement metric
  - `days_per_stream`: Inverse engagement indicator
  - `streams_per_session`: Activity intensity metric
  - `user_tenure_month`: User loyalty metric
  - `support_intensity`: Customer support usage indicator

### 3. Modeling & Evaluation
- **Baseline Models:**
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  
- **Hyperparameter Tuning:**
  - Grid search with 5-fold cross-validation
  - Optimized Gradient Boosting parameters
  - Feature importance analysis

### 4. Visualizations
- Churn distribution and demographics
- Feature correlation heatmaps
- Model performance metrics
- ROC curves and confusion matrices
- Feature importance rankings

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HuzaibShafi/Spotify-Analysis-Dataset-2025.git
cd Spotify-Analysis-Dataset-2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

Simply run the main analysis script:

```bash
python spotify_analysis.py
```

This will:
1. Load data from `data_repo/` (or generate sample data if none exists)
2. Perform comprehensive EDA
3. Clean and engineer features
4. Train baseline models
5. Perform hyperparameter tuning
6. Generate all visualizations

### Adding Your Own Data

To use your own Spotify dataset:

1. Place your CSV/JSON/Excel file in the `data_repo/` directory
2. Ensure your data includes:
   - User identification
   - Listening metrics (streams, sessions, duration)
   - User demographics
   - A target `churn` column (0/1 or True/False)

3. Run the script:
```bash
python spotify_analysis.py
```

## 🔬 Pipeline Steps

### Step 1: Data Loading
- Automatically detects and loads data files (CSV, JSON, Excel)
- Generates sample data if none found (for demonstration)
- Reports dataset shape and basic statistics

### Step 2: Exploratory Data Analysis
- Dataset information and missing value analysis
- Target variable distribution
- Statistical summaries
- Comprehensive visualizations

### Step 3: Data Cleaning & Feature Engineering
- Remove duplicates
- Handle missing values (median for numeric, mode for categorical)
- Create new engineered features
- Handle outliers using IQR method
- Encode categorical variables

### Step 4: Baseline Model Training
- Train-test split (80/20)
- Train multiple baseline models
- Evaluate using AUC score
- Select best-performing model
- Generate detailed evaluation reports

### Step 5: Hyperparameter Tuning
- Grid search over parameter space
- 5-fold cross-validation
- Best parameter selection
- Final model evaluation
- Feature importance analysis

## 📊 Results

### Sample Run Results

With the generated sample dataset (10,000 users):

**Churn Rate:** 22.23%

**Baseline Model Performance:**
- **Logistic Regression:** AUC = 0.7154 (Best)
- **Random Forest:** AUC = 0.6980
- **Gradient Boosting:** AUC = 0.7130

**Tuned Model Performance:**
- **Gradient Boosting (Tuned):** Test AUC = 0.7179
- **Best Parameters:** 
  - `n_estimators`: 300
  - `learning_rate`: 0.01
  - `max_depth`: 3
  - `min_samples_split`: 10

**Top Important Features:**
1. `subscription_type` (73.77%)
2. `user_tenure_month` (3.54%)
3. `customer_support_contacts` (3.27%)
4. `days_since_registration` (3.03%)
5. `support_intensity` (2.99%)

### Key Insights

1. **Subscription Type** is the strongest predictor of churn
2. **User Tenure** and **Support Contacts** are important indicators
3. **Engagement metrics** (streams, sessions) significantly impact churn
4. Free tier users have higher churn probability
5. Users with frequent support contacts are at higher risk

## 📁 Project Structure

```
Spotify-Analysis-Dataset-2025/
│
├── spotify_analysis.py          # Main analysis pipeline
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── data_repo/                    # Data directory
│   └── (your data files here)
│
└── outputs/                      # Generated outputs
    ├── plots/                    # All visualizations
    │   ├── eda_overview.png
    │   ├── correlation_matrix.png
    │   ├── metrics_by_churn.png
    │   ├── logistic_regression_evaluation.png
    │   └── feature_importance_tuned.png
    ├── models/                   # Saved models
    └── cleaned_data.csv          # Processed dataset
```

## 📈 Generated Visualizations

The pipeline generates several key visualizations:

1. **EDA Overview** (`eda_overview.png`)
   - Churn distribution
   - Churn by subscription type
   - Age distribution by churn
   - Platform-specific churn rates

2. **Correlation Matrix** (`correlation_matrix.png`)
   - Feature correlation heatmap

3. **Metrics by Churn** (`metrics_by_churn.png`)
   - Box plots comparing key metrics between churned and non-churned users

4. **Model Evaluation** (`logistic_regression_evaluation.png`)
   - Confusion matrix
   - ROC curve

5. **Feature Importance** (`feature_importance_tuned.png`)
   - Top 15 most important features for prediction

## 🛠️ Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new models
- Improve feature engineering
- Enhance visualizations
- Add tests

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Huzaib Shafi**
- GitHub: [@HuzaibShafi](https://github.com/HuzaibShafi)

## 🙏 Acknowledgments

- Spotify for the inspiration
- The open-source ML community

---

**Note:** This project uses sample data generation when no dataset is provided. Replace the data in `data_repo/` with your actual Spotify data for production use.
