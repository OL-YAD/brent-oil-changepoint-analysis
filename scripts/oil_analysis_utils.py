# oil_analysis_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Brent oil price data
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame
    """
    # Read data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set date as index
    #df.set_index('Date', inplace=True)
    
    # Sort index
    df.sort_index(inplace=True)
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    # Check missing values
    missing_values = df.isnull().sum()
    
    if missing_values.sum() > 0:
        # Interpolate missing values
        df = df.interpolate(method='time')
        # Forward fill any remaining missing values
        df = df.ffill()
    
    return df, missing_values

def add_features(df):
    """
    Add technical features to the dataset
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: DataFrame with additional features
    """
    # Calculate returns
    df['Returns'] = df['Price'].pct_change()
    
    # Calculate volatility (30-day rolling standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=30).std()
    
    # Add moving averages
    df['MA_50'] = df['Price'].rolling(window=50).mean()
    df['MA_200'] = df['Price'].rolling(window=200).mean()
    
    # Calculate price momentum
    df['Momentum'] = df['Price'].pct_change(periods=20)
    
    # Calculate log returns
    df['Log_Returns'] = np.log(df['Price']/df['Price'].shift(1))
    
    return df

def check_stationarity(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Parameters:
    series (pd.Series): Time series to test
    
    Returns:
    dict: Dictionary containing test results
    """
    result = adfuller(series.dropna())
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical values': result[4]
    }

def plot_time_series(df):
    """
    Create time series plot with moving averages
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['Price'], label='Price')
    plt.plot(df.index, df['MA_50'], label='50-day MA')
    plt.plot(df.index, df['MA_200'], label='200-day MA')
    plt.title('Brent Oil Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_returns_distribution(df):
    """
    Plot distribution of returns
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Returns'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_volatility(df):
    """
    Plot volatility over time
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Volatility'])
    plt.title('30-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

def analyze_autocorrelation(series, lags=40):
    """
    Plot ACF and PACF
    
    Parameters:
    series (pd.Series): Time series to analyze
    lags (int): Number of lags to plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function')
    
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()

def generate_summary_statistics(df):
    """
    Generate summary statistics for the dataset
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats_dict = {
        'date_range': {
            'start': df.index.min(),
            'end': df.index.max(),
            'total_days': len(df)
        },
        'price_statistics': {
            'mean': df['Price'].mean(),
            'std': df['Price'].std(),
            'min': df['Price'].min(),
            'max': df['Price'].max(),
            'median': df['Price'].median()
        },
        'returns_statistics': {
            'mean': df['Returns'].mean(),
            'std': df['Returns'].std(),
            'skewness': stats.skew(df['Returns'].dropna()),
            'kurtosis': stats.kurtosis(df['Returns'].dropna())
        },
        'volatility_statistics': {
            'mean': df['Volatility'].dropna().mean(),
            'std': df['Volatility'].dropna().std(),
            'max': df['Volatility'].dropna().max()
        }
    }
    
    return stats_dict



import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
import ruptures
import pywt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_distribution(series):
    """
    Perform comprehensive distribution analysis
    """
    results = {
        'normality': {
            'shapiro': stats.shapiro(series),
            'jarque_bera': stats.jarque_bera(series)
        },
        'descriptive': {
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series)
        },
        'stationarity': adfuller(series)
    }
    return results

def detect_structural_breaks(series, min_size=60):
    """
    Detect structural breaks using various methods
    """
    # Ruptures library for change point detection
    algo = ruptures.Pelt(model="rbf").fit(series.values.reshape(-1, 1))
    change_points = algo.predict(pen=10)
    
    return {
        'change_points': change_points,
        'n_changes': len(change_points)
    }

def analyze_seasonality(series, period=252):
    """
    Analyze seasonal patterns in the time series
    """
    decomposition = sm.tsa.seasonal_decompose(
        series, 
        period=period,
        extrapolate_trend='freq'
    )
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }

def calculate_risk_metrics(returns, alpha=0.05):
    """
    Calculate various risk metrics
    """
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    
    return {
        'VaR': var,
        'CVaR': cvar,
        'volatility': returns.std() * np.sqrt(252),
        'downside_risk': returns[returns < 0].std() * np.sqrt(252)
    }

def analyze_market_efficiency(series):
    """
    Perform market efficiency tests
    """
    # Variance ratio test
    vr_test = sm.stats.diagnostic.variance_ratio(series)
    
    # Runs test
    median = np.median(series)
    runs = np.sum(np.abs(np.diff(series > median))) + 1
    
    return {
        'variance_ratio': vr_test,
        'runs_test': runs,
        'autocorr_test': acorr_ljungbox(series)
    }

def plot_advanced_charts(df):
    """
    Create advanced visualization charts
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price and volume
    axes[0].plot(df.index, df['Price'])
    axes[0].set_title('Price Evolution')
    
    # Returns distribution
    sns.histplot(df['Returns'], kde=True, ax=axes[1])
    axes[1].set_title('Returns Distribution')
    
    # Rolling volatility
    axes[2].plot(df.index, df['Volatility'])
    axes[2].set_title('Rolling Volatility')
    
    plt.tight_layout()
    return fig

def validate_data_quality(df):
    """
    Perform comprehensive data quality checks
    """
    checks = {
        'missing_values': df.isnull().sum(),
        'duplicates': df.index.duplicated().sum(),
        'outliers': {
            col: detect_outliers(df[col]) 
            for col in df.columns if df[col].dtype in ['float64', 'int64']
        }
    }
    return checks

def detect_outliers(series, n_std=3):
    """
    Detect outliers using various methods
    """
    z_scores = np.abs(stats.zscore(series))
    iqr = stats.iqr(series)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    
    return {
        'z_score_outliers': np.sum(z_scores > n_std),
        'iqr_outliers': np.sum((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr)))
    }