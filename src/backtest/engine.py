"""
Backtesting engine module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from tqdm import tqdm

class BacktestEngine:
    """Implements walk-forward backtesting with realistic assumptions"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def run(self, price_data, weights_generator, benchmark_data=None):
        """Run walk-forward backtest"""
        logger.info("Starting walk-forward backtest...")
        
        # Initialize results containers
        equity_curve = pd.Series(index=price_data.index, dtype=float)
        weights_history = pd.DataFrame(index=price_data.index, 
                                       columns=price_data.columns)
        trades_history = []
        
        # Set initial capital
        capital = self.config.INITIAL_CAPITAL
        
        # Walk-forward windows
        train_days = self.config.TRAIN_WINDOW_DAYS
        test_days = self.config.TEST_WINDOW_DAYS
        step_days = self.config.STEP_DAYS
        
        dates = price_data.index
        
        # Progress bar
        pbar = tqdm(total=len(dates), desc="Backtesting")
        
        current_idx = train_days
        
        while current_idx < len(dates):
            # Define train and test periods
            train_start = dates[current_idx - train_days]
            train_end = dates[current_idx - 1]
            test_start = dates[current_idx]
            test_end_idx = min(current_idx + test_days, len(dates) - 1)
            test_end = dates[test_end_idx]
            
            # Get training data
            train_prices = price_data.loc[train_start:train_end]
            
            # Generate signals and weights for training period
            # In practice, this would use the signal generator
            train_returns = train_prices.pct_change().dropna()
            
            # Simple momentum-based weights for demonstration
            recent_returns = train_returns.iloc[-60:].mean()
            weights = self.simple_momentum_weights(recent_returns)
            
            # Apply constraints
            weights = self.apply_backtest_constraints(weights)
            
            # Execute test period
            test_prices = price_data.loc[test_start:test_end]
            test_period_returns = self.execute_test_period(
                test_prices, weights, capital
            )
            
            # Update equity curve
            equity_curve.loc[test_start:test_end] = test_period_returns
            
            # Update weights history
            weights_history.loc[test_start] = weights
            
            # Update capital
            cumulative_return = (1 + test_period_returns).prod() - 1
            capital *= (1 + cumulative_return)
            
            # Move forward
            current_idx += step_days
            pbar.update(step_days)
        
        pbar.close()
        
        # Calculate metrics
        metrics = self.calculate_metrics(equity_curve, benchmark_data)
        
        # Prepare results
        results = {
            'equity_curve': equity_curve,
            'weights_history': weights_history.dropna(),
            'metrics': metrics,
            'trades': trades_history
        }
        
        logger.info("Backtest completed successfully")
        return results
    
    def simple_momentum_weights(self, recent_returns):
        """Simple momentum strategy for demonstration"""
        # Rank stocks by recent returns
        ranked = recent_returns.rank()
        
        # Top 20% get positive weights, bottom 20% negative (for long-short)
        # For long-only, just take top 20%
        top_n = int(len(recent_returns) * 0.2)
        top_symbols = ranked.nlargest(top_n).index
        
        # Equal weight among top performers
        weights = pd.Series(0, index=recent_returns.index)
        weights[top_symbols] = 1 / top_n
        
        return weights
    
    def apply_backtest_constraints(self, weights):
        """Apply backtesting constraints"""
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Apply weight limits
        weights = weights.clip(
            lower=self.config.MIN_WEIGHT,
            upper=self.config.MAX_WEIGHT
        )
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
    
    def execute_test_period(self, test_prices, weights, capital):
        """Execute trades and track returns for test period"""
        # Calculate daily returns based on weights
        # Assumes daily rebalancing to target weights
        
        daily_returns = test_prices.pct_change().iloc[1:]  # Skip first NaN
        
        # Portfolio returns (weighted average of asset returns)
        portfolio_returns = (daily_returns * weights).sum(axis=1)
        
        # Apply transaction costs (simplified)
        # Assume rebalancing each day incurs costs
        turnover = np.abs(weights.diff().fillna(weights)).sum()
        transaction_cost = turnover * self.config.TRANSACTION_COST
        
        # Adjust returns for transaction costs
        portfolio_returns_adjusted = portfolio_returns - transaction_cost
        
        # Apply slippage
        portfolio_returns_adjusted = portfolio_returns_adjusted - self.config.SLIPPAGE
        
        return portfolio_returns_adjusted
    
    def calculate_metrics(self, portfolio_returns, benchmark_returns=None):
        """Calculate performance metrics"""
        from src.backtest.metrics import calculate_all_metrics
        
        metrics = calculate_all_metrics(
            portfolio_returns, 
            benchmark_returns,
            risk_free_rate=0.02/252  # Daily risk-free rate
        )
        
        return metrics
