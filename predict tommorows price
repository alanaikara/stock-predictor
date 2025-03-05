import pandas as pd
import numpy as np
import yfinance as yf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class NiftyTradingBot:
    def __init__(self, tickers=None):
        """
        Initialize the Nifty Trading Bot with top 50 Nifty stocks
        """
        if tickers is None:
            # Nifty 50 top stocks (NSE tickers)
            self.tickers = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS',
                'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 
                'ITC.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BAJFINANCE.NS', 
                'MARUTI.NS', 'SUNPHARMA.NS', 'BAJAJ-AUTO.NS', 'NTPC.NS', 
                'M&M.NS', 'POWERGRID.NS', 'WIPRO.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 
                'ADANIPORTS.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HEROMOTOCO.NS', 
                'NESTLEIND.NS', 'JSWSTEEL.NS', 'ASIANPAINT.NS', 'TATACONSUM.NS', 
                'BAJAJHLDNG.NS', 'APOLLOHOSP.NS', 'TATAMOTORS.NS', 'COALINDIA.NS', 
                'DRREDDY.NS', 'CIPLA.NS', 'TECHM.NS', 'UPL.NS', 'DIVISLAB.NS', 
                'INDUSINDBK.NS', 'ONGC.NS', 'HCLTECH.NS', 'TATASTEEL.NS', 
                'PIDILITIND.NS', 'BRITANNIA.NS', 'DLF.NS', 'SIEMENS.NS', 
                'ADANIPOWER.NS', 'DABUR.NS'
            ]
        else:
            self.tickers = tickers
        
        self.data = {}
        self.predictions = {}
    
    def fetch_historical_data(self, start_date='2020-01-01', end_date='2025-03-06'):
        """
        Fetch historical stock data for all Nifty tickers
        """
        time.sleep(2)  # Wait 2 seconds before fetching the next stock

        for ticker in self.tickers:
            try:
                # Use .NS suffix for NSE stocks
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                # Calculate advanced features
                stock_data['Returns'] = stock_data['Close'].pct_change()
                stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
                stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
                stock_data['Volatility'] = stock_data['Returns'].rolling(window=50).std() * np.sqrt(252)
                
                # Additional market-specific indicators
                stock_data['RSI'] = self._calculate_rsi(stock_data['Close'])
                
                # Drop NaN values
                stock_data.dropna(inplace=True)
                
                self.data[ticker] = stock_data
                print(f"Successfully fetched data for {ticker}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    
    def _calculate_rsi(self, prices, periods=14):
        """
        Calculate Relative Strength Index (RSI)
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def prepare_features(self, ticker):
        """
        Prepare features for machine learning model
        """
        df = self.data[ticker]
        
        # Select features with market-specific indicators
        features = [
            'Returns', 'SMA_50', 'EMA_20', 'Volatility', 
            'Volume', 'Open', 'High', 'Low', 'RSI'
        ]
        
        X = df[features]
        y = df['Close'].shift(-1)  # Predict next day's closing price
        
        # Drop NaN values after shifting
        X = X[:-1]
        y = y[:-1]
        
        return X, y
    
    def train_prediction_model(self, ticker):
        """
        Train Random Forest Regression model for stock prediction
        """
        X, y = self.prepare_features(ticker)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance for {ticker}:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")
        
        return model, scaler
    
    def predict_stock_performance(self):
        """
        Predict stock performance for all tickers
        """
        for ticker in self.tickers:
            try:
                model, scaler = self.train_prediction_model(ticker)
                
                # Get latest data for prediction
                latest_data = self.data[ticker].iloc[-1]
                features = [
                    'Returns', 'SMA_50', 'EMA_20', 'Volatility', 
                    'Volume', 'Open', 'High', 'Low', 'RSI'
                ]
                
                latest_features = latest_data[features].values.reshape(1, -1)
                latest_features_scaled = scaler.transform(latest_features)
                
                prediction = model.predict(latest_features_scaled)[0]
                self.predictions[ticker] = {
                    'predicted_price': prediction,
                    'current_price': latest_data['Close']
                }
            except Exception as e:
                print(f"Error predicting for {ticker}: {e}")
    
    def analyze_predictions(self):
        """
        Analyze and rank stock predictions with additional insights
        """
        prediction_analysis = []
        for ticker, pred_data in self.predictions.items():
            try:
                current_price = float(pred_data['current_price'])
                predicted_price = float(pred_data['predicted_price'])
                
                # Calculate potential return
                potential_return = ((predicted_price - current_price) / current_price) * 100
                
                # Calculate risk (using volatility from historical data)
                volatility = float(self.data[ticker]['Returns'].std() * np.sqrt(252) * 100)
                
                # Avoid division by zero
                risk_adjusted_return = potential_return / volatility if volatility != 0 else 0
                
                prediction_analysis.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'potential_return': potential_return,
                    'volatility': volatility,
                    'risk_adjusted_return': risk_adjusted_return
                })
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
        
        # Convert to DataFrame to ensure consistent handling
        df_predictions = pd.DataFrame(prediction_analysis)
        
        # Sort by risk-adjusted return
        df_predictions_sorted = df_predictions.sort_values('risk_adjusted_return', ascending=False)
        
        print("\nNifty Stock Prediction Rankings:")
        print("Ranking considers potential return adjusted for volatility")
        for rank, row in df_predictions_sorted.head(10).iterrows():
            print(f"{rank+1}. {row['ticker']}: ")
            print(f"   Potential Return: {row['potential_return']:.2f}%")
            print(f"   Volatility: {row['volatility']:.2f}%")
            print(f"   Risk-Adjusted Return: {row['risk_adjusted_return']:.2f}")
        
        return df_predictions_sorted

# Usage example
def main():
    bot = NiftyTradingBot()
    bot.fetch_historical_data()
    bot.predict_stock_performance()
    recommended_stocks = bot.analyze_predictions()
    recommended_stocks.to_csv('recommended_stocks.csv', index=False)

if __name__ == "__main__":
    main()