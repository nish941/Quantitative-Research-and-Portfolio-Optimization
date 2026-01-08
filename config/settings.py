"""
Configuration settings for the portfolio optimization system
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any
import yaml

@dataclass
class Config:
    """Main configuration class"""
    
    # Data settings
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2024-01-01"
    REBALANCE_FREQUENCY: str = "W"  # Weekly rebalancing
    
    # Universe settings
    UNIVERSE_SYMBOLS: List[str] = None
    BENCHMARK_SYMBOL: str = "^GSPC"  # S&P 500
    RISK_FREE_RATE_SYMBOL: str = "^TNX"  # 10-Year Treasury
    
    # Signal generation parameters
    MOMENTUM_WINDOWS: List[int] = (20, 60, 120)  # Days
    VOLATILITY_WINDOW: int = 60
    MIN_VOLATILITY: float = 0.15  # Minimum annualized volatility
    MAX_VOLATILITY: float = 0.60  # Maximum annualized volatility
    
    # Portfolio optimization
    TARGET_VOLATILITY: float = 0.15
    MAX_WEIGHT: float = 0.10  # Maximum 10% per position
    MIN_WEIGHT: float = 0.01  # Minimum 1% per position
    TURNOVER_TARGET: float = 0.20  # 20% max turnover
    
    # Backtesting
    INITIAL_CAPITAL: float = 1000000.0
    TRANSACTION_COST: float = 0.001  # 10 bps per trade
    SLIPPAGE: float = 0.0005  # 5 bps slippage
    
    # Risk management
    MAX_DRAWDOWN_LIMIT: float = 0.15  # 15% max drawdown
    VAR_CONFIDENCE: float = 0.95
    BETA_TARGET: float = 1.0
    BETA_TOLERANCE: float = 0.1
    
    # Walk-forward parameters
    TRAIN_WINDOW_DAYS: int = 252 * 3  # 3 years
    TEST_WINDOW_DAYS: int = 252  # 1 year
    STEP_DAYS: int = 63  # Quarterly steps
    
    def __post_init__(self):
        """Load universe symbols from config file"""
        if self.UNIVERSE_SYMBOLS is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                'universe_config.yaml'
            )
            with open(config_path, 'r') as f:
                universe_config = yaml.safe_load(f)
                self.UNIVERSE_SYMBOLS = universe_config['universe_symbols']
