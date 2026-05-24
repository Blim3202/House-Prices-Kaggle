# src/utils.py
## DISCLAIMER: These helper funtions are taken from DeepSeek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(data_path: str = "data/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission data."""
    train = pd.read_csv(f"{data_path}/train.csv")
    test = pd.read_csv(f"{data_path}/test.csv")
    sample = pd.read_csv(f"{data_path}/sample_submission.csv")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {sample.shape}")
    return train, test, sample

def plot_sale_price_distribution(train_df: pd.DataFrame, save_path: str = None):
    """Plot distribution of SalePrice before and after log transform."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original distribution
    axes[0].hist(train_df['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Original SalePrice Distribution')
    axes[0].set_xlabel('SalePrice')
    axes[0].set_ylabel('Frequency')
    
    # Log transformed
    log_price = np.log1p(train_df['SalePrice'])
    axes[1].hist(log_price, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title('Log-Transformed SalePrice')
    axes[1].set_xlabel('log(SalePrice)')
    axes[1].set_ylabel('Frequency')
    
    # Box plot to see outliers
    axes[2].boxplot(train_df['SalePrice'], vert=True)
    axes[2].set_title('Box Plot of SalePrice')
    axes[2].set_ylabel('SalePrice')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Skewness (original): {train_df['SalePrice'].skew():.3f}")
    print(f"Skewness (log transformed): {log_price.skew():.3f}")
    return log_price

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame showing missing value statistics."""
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
    return missing_df

def identify_na_as_absence_columns() -> List[str]:
    """
    Columns where NA means "absence of feature" (e.g., no garage, no basement).
    Based on data_description.txt.
    Bryce: 24/05/2026 - Confirmed that these are the only features with NA = absence. Compared with data_description.txt and searched for NA with case sensitivity.
    """
    return [
        'Alley',           # NA = No alley access
        'BsmtQual',        # NA = No basement
        'BsmtCond',        # NA = No basement
        'BsmtExposure',    # NA = No basement
        'BsmtFinType1',    # NA = No basement
        'BsmtFinType2',    # NA = No basement
        'FireplaceQu',     # NA = No fireplace
        'GarageType',      # NA = No garage
        'GarageFinish',    # NA = No garage
        'GarageQual',      # NA = No garage
        'GarageCond',      # NA = No garage
        'PoolQC',          # NA = No pool
        'Fence',           # NA = No fence
        'MiscFeature'      # NA = No miscellaneous feature
    ]

def replace_na_with_none(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Replace NA with 'None' for columns where NA indicates absence."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    return df_clean

def numeric_vs_categorical_split(df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical."""
    if exclude_cols is None:
        exclude_cols = ['Id', 'SalePrice']
    
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, categorical_cols

def plot_correlation_with_target(df: pd.DataFrame, target: str = 'SalePrice', top_n: int = 15, save_path: str = None):
    """Plot correlation of numeric features with target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    correlations = df[numeric_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n)
    
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh')
    plt.xlabel(f'Absolute Correlation with {target}')
    plt.title(f'Top {top_n} Numeric Features Correlated with {target}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    
    return top_features

def plot_categorical_impact(df: pd.DataFrame, categorical_cols: List[str], target: str = 'SalePrice', max_categories: int = 10, save_path: str = None):
    """Box plots showing impact of categorical features on target."""
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for idx, col in enumerate(categorical_cols[:9]):  # Limit to 9 for visibility
        if df[col].nunique() <= max_categories:
            df.groupby(col)[target].median().sort_values().plot(kind='barh', ax=axes[idx])
            axes[idx].set_title(f'Median {target} by {col}')
            axes[idx].set_xlabel(f'Median {target}')
    
    # Hide unused subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()