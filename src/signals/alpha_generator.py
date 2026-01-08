"""
Alpha signal generation module
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from loguru import logger

class AlphaGenerator:
    """Generates alpha signals using multiple methodologies"""
    
    def __init__(self, config):
        self.config = config
        
    def generate_all_signals(self, price_data):
        """Generate all alpha signals"""
        logger.info("Generating alpha signals...")
        
        signals = {}
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # 1. Momentum signals
        signals['momentum'] = self.momentum_signals(returns)
        
        # 2. Mean reversion signals
        signals['mean_reversion'] = self.mean_reversion_signals(returns)
        
        # 3. Volatility signals
        signals['volatility'] = self.volatility_signals(returns)
        
        # 4. Cross-sectional signals
        signals['cross_sectional'] = self.cross_sectional_signals(returns)
        
        # 5. Beta-adjusted signals
        signals['beta_adjusted'] = self.beta_adjusted_signals(returns, price_data)
        
        # Combine signals with weights
        combined_signals = self.combine_signals(signals)
        
        # Apply filters
        filtered_signals = self.apply_filters(combined_signals, returns)
        
        logger.info(f"Generated {len(filtered_signals.columns)} signals")
        return filtered_signals
    
    def momentum_signals(self, returns):
        """Generate momentum-based signals"""
        signals = pd.DataFrame(index=returns.index)
        
        for window in self.config.MOMENTUM_WINDOWS:
            # Calculate momentum
            momentum = returns.rolling(window).mean()
            
            # Z-score normalization
            signals[f'momentum_{window}'] = (
                momentum - momentum.rolling(252).mean()
            ) / momentum.rolling(252).std()
        
        return signals
    
    def mean_reversion_signals(self, returns):
        """Generate mean reversion signals"""
        signals = pd.DataFrame(index=returns.index)
        
        # RSI-like signal
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gain = gains.rolling(14).mean()
        avg_loss = losses.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize RSI to [-1, 1]
        signals['mean_reversion'] = (rsi - 50) / 50
        
        return signals
    
    def volatility_signals(self, returns):
        """Generate volatility-based signals"""
        signals = pd.DataFrame(index=returns.index)
        
        # Calculate rolling volatility
        vol_window = self.config.VOLATILITY_WINDOW
        volatility = returns.rolling(vol_window).std() * np.sqrt(252)
        
        # Volatility regime signal
        vol_regime = (
            volatility - volatility.rolling(252).mean()
        ) / volatility.rolling(252).std()
        
        # Inverse volatility signal (low vol -> higher weight)
        inv_vol = 1 / volatility
        inv_vol = (inv_vol - inv_vol.rolling(252).mean()) / inv_vol.rolling(252).std()
        
        signals['vol_regime'] = vol_regime
        signals['inv_vol'] = inv_vol
        
        return signals
    
    def cross_sectional_signals(self, returns):
        """Generate cross-sectional ranking signals"""
        signals = pd.DataFrame(index=returns.index)
        
        # Rank by recent performance
        for window in [5, 20, 60]:
            recent_returns = returns.rolling(window).mean()
            
            # Cross-sectional rank (0 to 1)
            rank = recent_returns.rank(axis=1, pct=True)
            
            # Normalize to [-1, 1]
            signals[f'rank_{window}'] = (rank - 0.5) * 2
        
        return signals
    
    def beta_adjusted_signals(self, returns, prices):
        """Generate beta-adjusted signals"""
        signals = pd.DataFrame(index=returns.index)
        
        # Calculate beta to benchmark (simplified - in practice use proper benchmark)
        # Here we use first principal component as market proxy
        try:
            # PCA to extract market factor
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            market_factor = pd.Series(
                pca.fit_transform(returns.fillna(0))[:, 0],
                index=returns.index
            )
            
            # Calculate rolling beta
            beta_window = 60
            betas = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for symbol in returns.columns:
                # Rolling regression
                for i in range(beta_window, len(returns)):
                    window_returns = returns.iloc[i-beta_window:i]
                    window_market = market_factor.iloc[i-beta_window:i]
                    
                    if window_returns[symbol].notna().all():
                        X = window_market.values.reshape(-1, 1)
                        y = window_returns[symbol].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        betas.loc[returns.index[i], symbol] = model.coef_[0]
            
            # Beta-adjusted momentum
            momentum = returns.rolling(60).mean()
            beta_adjusted_momentum = momentum / betas
            
            signals['beta_adjusted'] = (
                beta_adjusted_momentum - beta_adjusted_momentum.rolling(252).mean()
            ) / beta_adjusted_momentum.rolling(252).std()
            
        except Exception as e:
            logger.warning(f"Could not compute beta-adjusted signals: {e}")
            signals['beta_adjusted'] = 0
        
        return signals
    
    def combine_signals(self, signal_dict):
        """Combine multiple signals with weights"""
        weights = {
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'volatility': 0.25,
            'cross_sectional': 0.15,
            'beta_adjusted': 0.1
        }
        
        combined = pd.DataFrame(index=signal_dict['momentum'].index)
        
        for signal_type, weight in weights.items():
            if signal_type in signal_dict:
                # Average across all columns of this signal type
                signal_avg = signal_dict[signal_type].mean(axis=1)
                combined[signal_type] = signal_avg * weight
        
        # Sum weighted signals
        combined['combined_signal'] = combined.sum(axis=1)
        
        # Cap extreme values
        combined['combined_signal'] = np.clip(
            combined['combined_signal'], -3, 3
        )
        
        return combined['combined_signal']
    
    def apply_filters(self, signals, returns):
        """Apply filters to signals"""
        filtered = signals.copy()
        
        # Filter by volatility
        volatility = returns.std() * np.sqrt(252)
        vol_filter = (
            (volatility >= self.config.MIN_VOLATILITY) &
            (volatility <= self.config.MAX_VOLATILITY)
        )
        
        # Only keep symbols passing volatility filter
        filtered = filtered.loc[:, vol_filter]
        
        # Filter by liquidity (average volume)
        # Note: In practice, you would use actual volume data
        
        return filtered
