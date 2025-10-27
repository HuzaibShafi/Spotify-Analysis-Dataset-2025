"""
Spotify Churn Analysis Pipeline
Author: Spotify Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modeling imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create directories for outputs
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)

# ==================== 
# 1. DATA LOADING
# ====================

def load_data(data_path="data_repo"):
    """
    Load Spotify dataset from the repository
    
    Args:
        data_path: Path to data directory
    
    Returns:
        DataFrame: Spotify data
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    # Look for data files
    data_files = list(Path(data_path).glob("*.csv")) + \
                 list(Path(data_path).glob("*.json")) + \
                 list(Path(data_path).glob("*.xlsx"))
    
    if not data_files:
        print(f"⚠️  No data files found in {data_path}")
        print("Generating sample data for demonstration...")
        return generate_sample_data()
    
    # Load the first available data file
    file_path = data_files[0]
    print(f"📁 Loading data from: {file_path}")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        df = pd.read_json(file_path)
    elif file_path.suffix == '.xlsx':
        df = pd.read_excel(file_path)
    
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def generate_sample_data():
    """Generate sample Spotify data for demonstration"""
    np.random.seed(42)
    
    n_samples = 10000
    
    data = {
        'user_id': range(1, n_samples + 1),
        'subscription_type': np.random.choice(['Free', 'Premium'], n_samples, p=[0.6, 0.4]),
        'age': np.random.randint(16, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.5, 0.45, 0.05]),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR'], n_samples),
        
        # Music listening metrics
        'total_streams': np.random.poisson(850, n_samples),
        'total_minutes_listened': np.random.normal(4500, 1200, n_samples).astype(int),
        'sessions_per_week': np.random.poisson(25, n_samples),
        'avg_session_duration': np.random.normal(45, 15, n_samples),
        'unique_artists': np.random.poisson(85, n_samples),
        'unique_songs': np.random.poisson(200, n_samples),
        
        # Platform metrics
        'app_version': np.random.choice(['6.0', '6.1', '6.2', '6.3'], n_samples),
        'platform': np.random.choice(['iOS', 'Android', 'Web', 'Desktop'], n_samples, p=[0.4, 0.35, 0.15, 0.1]),
        
        # Engagement metrics
        'playlist_created': np.random.poisson(15, n_samples),
        'share_count': np.random.poisson(5, n_samples),
        'follower_count': np.random.poisson(120, n_samples),
        
        # Support/Support metrics
        'customer_support_contacts': np.random.poisson(0.5, n_samples),
        'days_since_registration': np.random.normal(450, 180, n_samples).astype(int),
        
        # Features
        'ad_supported_listening': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'podcast_listening': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable: churn
    # Higher churn for: Free users, fewer streams, support contacts, older app versions
    churn_prob = (
        0.3 * (df['subscription_type'] == 'Free') +
        0.2 * (df['total_streams'] < 500) +
        0.15 * (df['customer_support_contacts'] > 1) +
        0.1 * (df['sessions_per_week'] < 10) +
        0.1 * (df['days_since_registration'] > 600) +
        np.random.normal(0, 0.1, n_samples)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    df['churn'] = np.random.binomial(1, churn_prob)
    
    print(f"✅ Generated sample data with {len(df)} rows and {len(df.columns)} columns")
    print(f"📊 Churn rate: {df['churn'].mean():.2%}")
    return df


# ==================== 
# 2. EXPLORATORY DATA ANALYSIS
# ====================

def exploratory_data_analysis(df):
    """
    Perform comprehensive exploratory data analysis
    
    Args:
        df: DataFrame to analyze
    """
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)
    
    # Basic info
    print("\n📋 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if not missing_df.empty:
        print(missing_df)
    else:
        print("No missing values!")
    
    print("\n📊 Target Variable Distribution:")
    if 'churn' in df.columns:
        churn_counts = df['churn'].value_counts()
        print(churn_counts)
        print(f"\nChurn Rate: {df['churn'].mean():.2%}")
    else:
        print("No 'churn' column found")
    
    # Numerical summary
    print("\n📈 Numerical Summary:")
    print(df.describe())
    
    # Create visualizations
    create_eda_visualizations(df)
    

def create_eda_visualizations(df):
    """Create comprehensive EDA visualizations"""
    print("\n📊 Creating visualizations...")
    
    if 'churn' not in df.columns:
        print("⚠️  No 'churn' column found. Skipping churn-specific visualizations.")
        return
    
    # 1. Churn distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exploratory Data Analysis - Spotify Churn', fontsize=16, fontweight='bold')
    
    # Churn distribution
    df['churn'].value_counts().plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%', 
                                    colors=['#1DB954', '#f96b6b'])
    axes[0, 0].set_title('Churn Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('')
    
    # Subscription type vs churn
    if 'subscription_type' in df.columns:
        sns.countplot(data=df, x='subscription_type', hue='churn', ax=axes[0, 1])
        axes[0, 1].set_title('Churn by Subscription Type', fontsize=12, fontweight='bold')
    
    # Age distribution
    if 'age' in df.columns:
        sns.histplot(data=df, x='age', hue='churn', kde=True, ax=axes[0, 2])
        axes[0, 2].set_title('Age Distribution by Churn', fontsize=12, fontweight='bold')
    
    # Total streams
    if 'total_streams' in df.columns:
        sns.boxplot(data=df, x='churn', y='total_streams', ax=axes[1, 0])
        axes[1, 0].set_title('Total Streams by Churn', fontsize=12, fontweight='bold')
    
    # Sessions per week
    if 'sessions_per_week' in df.columns:
        sns.boxplot(data=df, x='churn', y='sessions_per_week', ax=axes[1, 1])
        axes[1, 1].set_title('Sessions per Week by Churn', fontsize=12, fontweight='bold')
    
    # Platform distribution
    if 'platform' in df.columns:
        churn_by_platform = pd.crosstab(df['platform'], df['churn'], normalize='index') * 100
        churn_by_platform.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Churn Rate by Platform', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('% Churn')
        axes[1, 2].legend(['No Churn', 'Churned'])
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/eda_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/plots/eda_overview.png")
    plt.close()
    
    # 2. Correlation heatmap
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        plt.figure(figsize=(14, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('outputs/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: outputs/plots/correlation_matrix.png")
        plt.close()
    
    # 3. Feature importance by churn (if continuous features exist)
    if 'total_streams' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Key Metrics by Churn Status', fontsize=16, fontweight='bold')
        
        metrics = [
            ('total_streams', 'Total Streams'),
            ('sessions_per_week', 'Sessions per Week'),
            ('unique_artists', 'Unique Artists'),
            ('total_minutes_listened', 'Total Minutes Listened')
        ]
        
        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            df.boxplot(column=metric, by='churn', ax=ax)
            ax.set_title(f'{label} by Churn', fontsize=12, fontweight='bold')
            ax.set_xlabel('Churn Status')
            ax.set_ylabel(label)
            plt.suptitle('')  # Remove default suptitle
        
        plt.tight_layout()
        plt.savefig('outputs/plots/metrics_by_churn.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: outputs/plots/metrics_by_churn.png")
        plt.close()


# ==================== 
# 3. DATA CLEANING & FEATURE ENGINEERING
# ====================

def clean_and_engineer_features(df):
    """
    Clean data and create features for churn modeling
    
    Args:
        df: Original DataFrame
    
    Returns:
        DataFrame: Cleaned and engineered DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 3: Data Cleaning & Feature Engineering")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"📊 Removed {initial_count - len(df_clean)} duplicate rows")
    
    # Handle missing values
    print("\n🧹 Handling missing values...")
    df_clean = handle_missing_values(df_clean)
    
    # Create new features
    print("\n⚙️ Creating new features...")
    df_clean = create_features(df_clean)
    
    # Handle outliers
    print("\n🎯 Handling outliers...")
    df_clean = handle_outliers(df_clean)
    
    # Categorical encoding
    print("\n🔤 Encoding categorical variables...")
    df_clean = encode_categorical(df_clean)
    
    print(f"✅ Cleaned dataset: {df_clean.shape}")
    return df_clean


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  → {col}: Filled with median")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  → {col}: Filled with mode")
    return df


def create_features(df):
    """Create engineered features"""
    # Engagement score (combination of multiple metrics)
    if all(col in df.columns for col in ['total_streams', 'sessions_per_week', 'unique_artists']):
        df['engagement_score'] = (
            df['total_streams'] / df['total_streams'].max() * 0.4 +
            df['sessions_per_week'] / df['sessions_per_week'].max() * 0.3 +
            df['unique_artists'] / df['unique_artists'].max() * 0.3
        )
        print("  ✓ Created: engagement_score")
    
    # Days per stream (inverse engagement metric)
    if 'days_since_registration' in df.columns and 'total_streams' in df.columns:
        df['days_per_stream'] = df['days_since_registration'] / (df['total_streams'] + 1)
        print("  ✓ Created: days_per_stream")
    
    # Streams per session
    if all(col in df.columns for col in ['total_streams', 'sessions_per_week']):
        df['streams_per_session'] = df['total_streams'] / (df['sessions_per_week'] + 1)
        print("  ✓ Created: streams_per_session")
    
    # User tenure (if not already exists)
    if 'days_since_registration' in df.columns:
        df['user_tenure_month'] = df['days_since_registration'] / 30
        print("  ✓ Created: user_tenure_month")
    
    # Churn risk indicator
    if 'customer_support_contacts' in df.columns:
        df['support_intensity'] = (df['customer_support_contacts'] > 1).astype(int)
        print("  ✓ Created: support_intensity")
    
    return df


def handle_outliers(df):
    """Handle outliers using IQR method for numerical columns"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col not in ['churn', 'ad_supported_listening', 'podcast_listening', 'support_intensity']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing rows
            outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
            if outliers_before > 0:
                print(f"  ✓ Capped outliers in {col}: {outliers_before} values adjusted")
    
    return df


def encode_categorical(df):
    """Encode categorical variables"""
    df_encoded = df.copy()
    
    # Use label encoding for categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ Encoded: {col}")
    
    return df_encoded


# ==================== 
# 4. MODEL TRAINING & EVALUATION
# ====================

def prepare_modeling_data(df):
    """
    Prepare data for modeling
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        X, y: Features and target
    """
    if 'churn' not in df.columns:
        raise ValueError("Target variable 'churn' not found in dataframe")
    
    # Exclude non-feature columns
    exclude_cols = ['churn', 'user_id'] if 'user_id' in df.columns else ['churn']
    X = df.drop(columns=exclude_cols)
    y = df['churn']
    
    return X, y


def train_baseline_models(X, y):
    """
    Train and evaluate baseline models
    
    Args:
        X: Features
        y: Target
    """
    print("\n" + "=" * 60)
    print("STEP 4: Baseline Model Training & Evaluation")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'auc': auc_score
        }
        
        print(f"✅ {name} AUC: {auc_score:.4f}")
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name}: AUC = {result['auc']:.4f}")
    
    # Best model
    best_model_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n🏆 Best Baseline Model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    
    # Detailed evaluation of best model
    evaluate_model(
        results[best_model_name]['model'],
        X_test_scaled,
        y_test,
        best_model_name
    )
    
    return results, best_model_name


def evaluate_model(model, X_test, y_test, model_name):
    """Detailed model evaluation"""
    print(f"\n📊 Detailed Evaluation: {model_name}")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name} - Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    # ROC curve
    axes[1].plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'outputs/plots/{model_name.replace(" ", "_").lower()}_evaluation.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: outputs/plots/{model_name.replace(' ', '_').lower()}_evaluation.png")
    plt.close()


# ==================== 
# 5. HYPERPARAMETER TUNING
# ====================

def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning for best model
    
    Args:
        X: Features
        y: Target
    """
    print("\n" + "=" * 60)
    print("STEP 5: Hyperparameter Tuning")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Best model based on baseline was Random Forest or Gradient Boosting
    # Let's tune Gradient Boosting
    print("\n🔧 Tuning Gradient Boosting Classifier...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Grid search with cross-validation
    print("⏳ Running grid search (this may take a while)...")
    grid_search = GridSearchCV(
        gb_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ Test set AUC: {test_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances - Tuned Gradient Boosting', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/plots/feature_importance_tuned.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: outputs/plots/feature_importance_tuned.png")
    plt.close()
    
    return best_model, grid_search.best_params_


# ==================== 
# MAIN EXECUTION
# ====================

def main():
    """Main pipeline execution"""
    print("\n" + "=" * 60)
    print("🎵 Spotify Churn Analysis Pipeline")
    print("=" * 60)
    
    try:
        # 1. Load data
        df = load_data()
        
        # 2. EDA
        exploratory_data_analysis(df)
        
        # 3. Clean and engineer features
        df_clean = clean_and_engineer_features(df)
        
        # Save cleaned data
        df_clean.to_csv('outputs/cleaned_data.csv', index=False)
        print(f"\n✅ Saved cleaned data: outputs/cleaned_data.csv")
        
        # 4. Train baseline models
        X, y = prepare_modeling_data(df_clean)
        results, best_model_name = train_baseline_models(X, y)
        
        # 5. Hyperparameter tuning
        tuned_model, best_params = hyperparameter_tuning(X, y)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Completed Successfully!")
        print("=" * 60)
        print("\n📁 Output files:")
        print("  → outputs/plots/ - All visualizations")
        print("  → outputs/models/ - Saved models")
        print("  → outputs/cleaned_data.csv - Processed dataset")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

