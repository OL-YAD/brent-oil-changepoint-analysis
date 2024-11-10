import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class OilPriceAnalysis:
    def __init__(self, data):
        """
        Initialize with DataFrame containing oil price data
        data: pandas DataFrame with DateTimeIndex
        """
        self.data = data
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def prepare_data(self, test_size=0.2):
        """Prepare data for modeling"""
        # Split data into train and test sets
        train_size = int(len(self.data) * (1 - test_size))
        self.train = self.data[:train_size]
        self.test = self.data[train_size:]
        
        # Scale data for LSTM
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data[['Price']])
        self.scaled_train = self.scaled_data[:train_size]
        self.scaled_test = self.scaled_data[train_size:]
        
    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM model"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def fit_arima(self, order=(1,1,1)):
        """Fit ARIMA model"""
        try:
            model = ARIMA(self.train['Price'], order=order)
            self.models['arima'] = model.fit()
            # Make predictions
            self.predictions['arima'] = self.models['arima'].forecast(steps=len(self.test))
            # Calculate metrics
            self.calculate_metrics('arima')
        except Exception as e:
            print(f"Error fitting ARIMA model: {str(e)}")
    
    def fit_garch(self, vol_order=(1,1)):
        """Fit GARCH model"""
        try:
            model = arch_model(self.train['Returns'], vol='Garch', p=vol_order[0], q=vol_order[1])
            self.models['garch'] = model.fit(disp='off')
            # Make predictions
            forecast = self.models['garch'].forecast(horizon=len(self.test))
            self.predictions['garch'] = forecast.variance.values[-1]
            # Calculate metrics for volatility predictions
            self.calculate_metrics('garch', target='Volatility')
        except Exception as e:
            print(f"Error fitting GARCH model: {str(e)}")
    
    def fit_var(self, maxlags=5):
        """Fit VAR model"""
        try:
            # Select features for VAR model
            features = ['Price', 'Returns', 'Volatility', 'Momentum']
            var_data = self.train[features]
            
            model = VAR(var_data)
            self.models['var'] = model.fit(maxlags=maxlags)
            
            # Make predictions
            lag_order = self.models['var'].k_ar
            forecast = self.models['var'].forecast(var_data.values[-lag_order:], steps=len(self.test))
            self.predictions['var'] = pd.DataFrame(forecast, columns=features, index=self.test.index)
            
            # Calculate metrics
            self.calculate_metrics('var')
        except Exception as e:
            print(f"Error fitting VAR model: {str(e)}")
    
    def fit_lstm(self, seq_length=60, epochs=100, batch_size=32):
        """Fit LSTM model"""
        try:
            # Prepare sequences
            X_train, y_train = self.create_sequences(self.scaled_train, seq_length)
            X_test, y_test = self.create_sequences(self.scaled_test, seq_length)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Fit model
            self.models['lstm'] = model
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.1, verbose=0)
            
            # Make predictions
            self.predictions['lstm'] = model.predict(X_test)
            # Inverse transform predictions
            self.predictions['lstm'] = self.scaler.inverse_transform(self.predictions['lstm'])
            
            # Calculate metrics
            self.calculate_metrics('lstm')
            
            return history
        except Exception as e:
            print(f"Error fitting LSTM model: {str(e)}")
    
    def calculate_metrics(self, model_name, target='Price'):
        """Calculate performance metrics for a model"""
        try:
            if model_name not in self.predictions:
                return
            
            true_values = self.test[target].values
            pred_values = self.predictions[model_name]
            
            if model_name == 'lstm':
                # Adjust true values for LSTM sequences
                true_values = true_values[60:]  # Remove first seq_length points
                
            self.metrics[model_name] = {
                'rmse': np.sqrt(mean_squared_error(true_values, pred_values)),
                'mae': mean_absolute_error(true_values, pred_values),
                'r2': r2_score(true_values, pred_values)
            }
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {str(e)}")
        
    def cross_validate(self, model_name, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(self.scaled_data):
            if model_name == 'lstm':
                # Prepare sequences
                X_train, y_train = self.create_sequences(self.scaled_data[train_idx], 60)
                X_test, y_test = self.create_sequences(self.scaled_data[test_idx], 60)
                
                # Train model
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(60, 1), return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                
                # Make predictions
                predictions = model.predict(X_test)
                score = np.sqrt(mean_squared_error(y_test, predictions))
            else:
                # Implement cross-validation for other models
                pass
            
            cv_scores.append(score)
        
        return np.mean(cv_scores), np.std(cv_scores)
    
    def plot_results(self, model_name):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        
        # Get the actual values
        actual_values = self.test['Price'].values
        predicted_values = self.predictions[model_name]
        
        if model_name == 'lstm':
            # For LSTM, we need to adjust the actual values to match predictions
            # Remove the first seq_length points from actual values
            actual_values = actual_values[60:]
            # Ensure test dates align with the predictions
            plot_dates = self.test.index[60:]
        else:
            plot_dates = self.test.index
        
        plt.plot(plot_dates, actual_values, label='Actual', color='blue')
        plt.plot(plot_dates, predicted_values, label=f'{model_name} Predictions', color='red')
        plt.title(f'{model_name} Model Predictions vs Actual Values')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
    def integrate_external_data(self, external_data):
        """
        Integrate external data sources
        external_data: DataFrame with same index as self.data
        """
        self.data = pd.concat([self.data, external_data], axis=1)
        # Recalculate features if necessary
        self.prepare_data()

