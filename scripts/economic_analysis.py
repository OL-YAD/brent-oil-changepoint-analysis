import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wbdata
import datetime
import logging
from typing import Dict, Optional
from scipy import stats



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define date range
start_date = datetime.datetime(1987, 5, 20)  
end_date = datetime.datetime(2022, 9, 30)   

# Define countries and indicators
countries = ['USA', 'SAU', 'RUS', 'IRN', 'CHN', 'ARE', 'IRQ', 'KWT', 'EUU']

indicators = {
    'NY.GDP.MKTP.CD': 'GDP Growth (%)',
    'FP.CPI.TOTL': 'Inflation Rate (%)',
    'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
    'PA.NUS.FCRF': 'Exchange Rate (Local Currency per USD)',
    'EG.FEC.RNEW.ZS': 'Renewable Energy Consumption (%)',
    'CC.ENTX.ENV.ZS': 'Environmental Tax Revenue (% of GDP)',
    'BN.GSR.GNFS.CD': 'Net Trade (BoP, current US$)',
    'EG.ELC.NGAS.ZS': 'Natural Gas Electricity Production (%)'
}

def fetch_indicator_data(indicator_code: str, indicator_name: str) -> Optional[pd.DataFrame]:
    """
    Fetch data for a specific indicator from World Bank
    """
    try:
        # Fetch data
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=countries)
        
        # Process data
        if data is not None and not data.empty:
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            logger.info(f"Successfully fetched {indicator_name} data")
            return data
        else:
            logger.warning(f"No data returned for {indicator_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching {indicator_name}: {str(e)}")
        return None

def fetch_all_indicators() -> Dict[str, pd.DataFrame]:
    """
    Fetch all economic indicators
    """
    indicator_data = {}
    
    for code, name in indicators.items():
        data = fetch_indicator_data(code, name)
        if data is not None:
            indicator_data[name] = data
            
    return indicator_data

def merge_indicators(indicator_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all indicator dataframes
    """
    try:
        # Start with the first dataframe
        merged_df = None
        
        for name, df in indicator_data.items():
            if merged_df is None:
                merged_df = df
                continue
                
            # Merge with existing dataframe
            merged_df = pd.merge(
                merged_df,
                df,
                on=['date', 'country'],
                how='outer'
            )
        
        if merged_df is not None:
            # Sort and clean the merged dataframe
            merged_df = merged_df.sort_values(['country', 'date'])
            merged_df = merged_df.fillna(method='ffill')
            logger.info("Successfully merged all indicators")
            return merged_df
        else:
            logger.warning("No data to merge")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error merging indicators: {str(e)}")
        return pd.DataFrame()

def analyze_correlations(merged_data: pd.DataFrame, oil_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between indicators and oil prices
    """
    try:
        # Merge with oil prices
        analysis_df = pd.merge(
            merged_data,
            oil_prices,
            on='date',
            how='inner'
        )
        
        # Calculate correlations for each country
        correlations = []
        for country in analysis_df['country'].unique():
            country_data = analysis_df[analysis_df['country'] == country]
            
            # Get correlations with oil price
            corr = country_data.corr()['Price'].drop(['Price'])
            
            corr_df = pd.DataFrame({
                'country': country,
                'indicator': corr.index,
                'correlation': corr.values
            })
            correlations.append(corr_df)
        
        return pd.concat(correlations, ignore_index=True)
        
    except Exception as e:
        logger.error(f"Error analyzing correlations: {str(e)}")
        return pd.DataFrame()

def plot_correlations(correlations: pd.DataFrame) -> None:
    """
    Create visualization for correlations
    """
    try:
        plt.figure(figsize=(15, 10))
        pivot_corr = correlations.pivot(
            index='indicator',
            #columns='country',
            values='correlation'
        )
        
        plt.imshow(pivot_corr, cmap='RdBu', aspect='auto')
        plt.colorbar(label='Correlation')
        
        plt.xticks(range(len(pivot_corr.columns)), pivot_corr.columns, rotation=45)
        plt.yticks(range(len(pivot_corr.index)), pivot_corr.index)
        
        plt.title('Correlation between Economic Indicators and Oil Prices by Country')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting correlations: {str(e)}")


def merge_with_oil_prices(merged_data: pd.DataFrame, oil_prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge economic indicators with oil prices, keeping only the Price column
    """
    try:
        # Ensure date columns are datetime
        oil_prices_df['Date'] = pd.to_datetime(oil_prices_df['Date'])
        
        # Create a year column for both dataframes
        merged_data['year'] = merged_data['Date'].dt.year
        oil_prices_df['year'] = oil_prices_df['Date'].dt.year
        
        # Keep only the Price column for merging
        oil_prices_yearly = oil_prices_df[['year', 'Price']].drop_duplicates()
        
        # Merge on year
        final_df = pd.merge(
            merged_data,
            oil_prices_yearly,
            on='year',
            how='inner'
        )
        
        return final_df
        
    except Exception as e:
        print(f"Error merging with oil prices: {str(e)}")
        return pd.DataFrame()


def create_visualizations(final_df: pd.DataFrame, oil_prices_df: pd.DataFrame):
    """
    Create various visualizations for the analysis
    """
    # 1. Time Series Plot of Oil Prices and Major Indicators
    plt.figure(figsize=(15, 8))
    for country in final_df['country'].unique():
        country_data = final_df[final_df['country'] == country]
        plt.plot(country_data['Date'], country_data['GDP Growth (%)'], label=f'{country} GDP Growth')
    
    plt.plot(oil_prices_df['Date'], oil_prices_df['Price'], 'r--', label='Oil Price', alpha=0.7)
    plt.title('GDP Growth and Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. Oil Price Distribution by Year
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=oil_prices_df, x=oil_prices_df['Date'].dt.year, y='Price')
    plt.title('Oil Price Distribution by Year')
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    # Exclude 'year' column for correlation
    numeric_cols = [col for col in numeric_cols if col != 'year']
    correlation_matrix = final_df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Economic Indicators and Oil Prices')
    plt.tight_layout()
    plt.show()

    # 4. Scatter Plots Matrix for Key Indicators
    key_indicators = ['GDP Growth (%)', 'Inflation Rate (%)', 'Price']
    sns.pairplot(final_df[key_indicators + ['country']], hue='country')
    plt.suptitle('Scatter Plot Matrix of Key Indicators', y=1.02)
    plt.show()

def perform_correlation_analysis(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform detailed correlation analysis between oil prices and economic indicators
    """
    correlation_results = []
    
    # Analyze for each country
    for country in final_df['country'].unique():
        country_data = final_df[final_df['country'] == country]
        
        # Get numeric columns excluding oil price statistics and year column
        indicator_cols = [col for col in country_data.select_dtypes(include=[np.number]).columns 
                         if not col.startswith('oil_price_') and col != 'year']
        
        for indicator in indicator_cols:
            # Calculate correlation
            correlation, p_value = stats.pearsonr(
                country_data[indicator].fillna(method='ffill'), 
                country_data['Price']
            )
            
            # Calculate rolling correlation (2-year window)
            rolling_corr = country_data[indicator].rolling(window=24).corr(country_data['Price'])
            
            correlation_results.append({
                'country': country,
                'indicator': indicator,
                'correlation': correlation,
                'abs_correlation': abs(correlation),
                'p_value': p_value,
                'significant': p_value < 0.05,
                'correlation_strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak',
                'avg_rolling_correlation': rolling_corr.mean()
            })
    
    return pd.DataFrame(correlation_results)

def plot_correlation_analysis(correlation_results: pd.DataFrame):
    """
    Visualize correlation analysis results
    """
    # 1. Bar plot of correlations by country and indicator
    plt.figure(figsize=(15, 8))
    pivot_data = correlation_results.pivot(index='indicator', columns='country', values='correlation')
    pivot_data.plot(kind='bar')
    plt.title('Correlation with Oil Prices by Country and Indicator')
    plt.xlabel('Economic Indicator')
    plt.ylabel('Correlation Coefficient')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. Significant correlations highlight
    plt.figure(figsize=(12, 6))
    significant_corr = correlation_results[correlation_results['significant']]
    sns.scatterplot(
        data=significant_corr, 
        x='correlation', 
        y='indicator', 
        hue='country', 
        size='avg_rolling_correlation',
        sizes=(50, 200)
    )
    plt.title('Significant Correlations with Oil Prices')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.show()