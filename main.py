#!/usr/bin/env python3
"""
Main execution script for Quantitative Portfolio Optimization
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import DataLoader
from src.signals.alpha_generator import AlphaGenerator
from src.portfolio.optimizer import PortfolioOptimizer
from src.backtest.engine import BacktestEngine
from src.visualization.performance_plots import PerformanceVisualizer
from config.settings import Config

def main():
    """Main execution pipeline"""
    
    # Initialize configuration
    config = Config()
    logger.add("logs/execution.log", rotation="1 day")
    logger.info("Starting Portfolio Optimization Pipeline")
    
    # Step 1: Load and process data
    logger.info("Step 1: Loading market data...")
    data_loader = DataLoader(config)
    price_data, benchmark_data = data_loader.load_all_data()
    
    # Step 2: Generate alpha signals
    logger.info("Step 2: Generating alpha signals...")
    alpha_gen = AlphaGenerator(config)
    signals = alpha_gen.generate_all_signals(price_data)
    
    # Step 3: Optimize portfolio
    logger.info("Step 3: Optimizing portfolio...")
    optimizer = PortfolioOptimizer(config)
    portfolio_weights = optimizer.optimize_portfolio(signals, price_data)
    
    # Step 4: Run backtest
    logger.info("Step 4: Running backtest...")
    backtester = BacktestEngine(config)
    results = backtester.run(
        price_data, 
        portfolio_weights,
        benchmark_data
    )
    
    # Step 5: Analyze performance
    logger.info("Step 5: Analyzing performance...")
    visualizer = PerformanceVisualizer(config)
    visualizer.create_all_plots(results)
    
    # Step 6: Generate reports
    logger.info("Step 6: Generating reports...")
    generate_reports(results)
    
    logger.info("Pipeline execution completed successfully!")
    return results

def generate_reports(results):
    """Generate performance and risk reports"""
    
    # Save performance metrics
    results['metrics'].to_csv('results/performance/metrics_summary.csv')
    
    # Save equity curve
    results['equity_curve'].to_csv('results/performance/equity_curves.csv')
    
    # Save portfolio weights history
    results['weights_history'].to_csv('results/portfolio/historical_weights.csv')
    
    logger.info(f"Reports saved to results/ directory")

if __name__ == "__main__":
    results = main()
