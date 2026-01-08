"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from loguru import logger

class DataLoader:
    """Handles loading and preprocessing of financial data"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        
    def load_all_data(self):
        """Load price data for universe and benchmarks"""
        logger.info(f"Loading data from {self.config.START_DATE} to {self.config.END_DATE}")
        
        # Load universe data
        universe_data = self.load_universe_data()
        
        # Load benchmark data
        benchmark_data = self.load_benchmark_data()
        
        # Load risk-free rate
        risk_free_data = self.load_risk_free_rate()
        
        # Clean and align data
        universe_data = self.clean_data(universe_data)
        benchmark_data = self.clean_data(benchmark_data)
        
        return {
            'prices': universe_data,
            'benchmark': benchmark_data,
            'risk_free': risk_free_data
        }
    
    def load_universe_data(self):
        """Load price data for all universe symbols"""
        all_data = {}
        
        for symbol in self.config.UNIVERSE_SYMBOLS:
            try:
                logger.debug(f"Loading data for {symbol}")
                data = yf.download(
                    symbol,
                    start=self.config.START_DATE,
                    end=self.config.END_DATE,
                    progress=False
                )
                all_data[symbol] = data['Adj Close']
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        # Combine into DataFrame
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        
        # Forward fill for missing data (up to 5 days)
        df = df.ffill(limit=5)
        
        # Drop symbols with more than 20% missing data
        missing_pct = df.isnull().sum() / len(df)
        valid_symbols = missing_pct[missing_pct < 0.2].index
        df = df[valid_symbols]
        
        logger.info(f"Loaded {len(valid_symbols)} symbols out of {len(self.config.UNIVERSE_SYMBOLS)}")
        return df
    
    def load_benchmark_data(self):
        """Load benchmark data"""
        logger.info(f"Loading benchmark data for {self.config.BENCHMARK_SYMBOL}")
        
        benchmark = yf.download(
            self.config.BENCHMARK_SYMBOL,
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            progress=False
        )
        
        benchmark_data = benchmark['Adj Close']
        benchmark_data.index = pd.to_datetime(benchmark_data.index)
        
        return benchmark_data
    
    def load_risk_free_rate(self):
        """Load risk-free rate data"""
        try:
            # Use 10-year treasury yield as proxy for risk-free rate
            treasury = yf.download(
                self.config.RISK_FREE_RATE_SYMBOL,
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False
            )
            
            # Convert annual yield to daily rate
            risk_free_rate = treasury['Adj Close'] / 100 / 252
            risk_free_rate = risk_free_rate.ffill()
            
            return risk_free_rate
            
        except:
            # Fallback to constant risk-free rate (2% annual)
            logger.warning("Using fallback risk-free rate of 2%")
            dates = pd.date_range(
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                freq='D'
            )
            return pd.Series(0.02/252, index=dates)
    
    def clean_data(self, data):
        """Clean and prepare data"""
        # Remove outliers (prices that changed more than 50% in a day)
        if isinstance(data, pd.DataFrame):
            returns = data.pct_change()
            outlier_mask = returns.abs() > 0.5
            data = data.where(~outlier_mask.shift(-1))
        
        # Interpolate remaining missing values
        data = data.interpolate(method='linear')
        
        return data
