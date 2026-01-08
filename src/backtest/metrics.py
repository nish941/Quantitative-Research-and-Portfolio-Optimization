"""
Performance metrics calculation module
"""

import pandas as pd
import numpy as np
from scipy import stats

def calculate_all_metrics(portfolio_returns, benchmark_returns=None, 
                         risk_free_rate=0.0):
    """Calculate comprehensive performance metrics"""
    
    metrics = {}
    
    # Basic return metrics
    metrics['total_return'] = calculate_total_return(portfolio_returns)
    metrics['annualized_return'] = calculate_annualized_return(portfolio_returns)
    metrics['annualized_volatility'] = calculate_annualized_volatility(portfolio_returns)
    
    # Risk-adjusted metrics
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(
        portfolio_returns, risk_free_rate
    )
    metrics['sortino_ratio'] = calculate_sortino_ratio(
        portfolio_returns, risk_free_rate
    )
    
    # Risk metrics
    metrics['max_drawdown'] = calculate_max_drawdown(portfolio_returns)
    metrics['calmar_ratio'] = calculate_calmar_ratio(portfolio_returns)
    metrics['var_95'] = calculate_value_at_risk(portfolio_returns, 0.95)
    metrics['cvar_95'] = calculate_conditional_var(portfolio_returns, 0.95)
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        aligned_returns, aligned_benchmark = align_series(
            portfolio_returns, benchmark_returns
        )
        
        metrics['alpha'] = calculate_alpha(
            aligned_returns, aligned_benchmark, risk_free_rate
        )
        metrics['beta'] = calculate_beta(
            aligned_returns, aligned_benchmark
        )
        metrics['tracking_error'] = calculate_tracking_error(
            aligned_returns, aligned_benchmark
        )
        metrics['information_ratio'] = calculate_information_ratio(
            aligned_returns, aligned_benchmark
        )
        
        # Up/Down capture ratios
        metrics['up_capture'] = calculate_up_capture(
            aligned_returns, aligned_benchmark
        )
        metrics['down_capture'] = calculate_down_capture(
            aligned_returns, aligned_benchmark
        )
    
    # Additional metrics
    metrics['win_rate'] = calculate_win_rate(portfolio_returns)
    metrics['profit_factor'] = calculate_profit_factor(portfolio_returns)
    metrics['skewness'] = portfolio_returns.skew()
    metrics['kurtosis'] = portfolio_returns.kurtosis()
    
    return pd.Series(metrics)

def calculate_total_return(returns):
    """Calculate total return"""
    return (1 + returns).prod() - 1

def calculate_annualized_return(returns):
    """Calculate annualized return"""
    total_return = calculate_total_return(returns)
    years = len(returns) / 252
    return (1 + total_return) ** (1 / years) - 1

def calculate_annualized_volatility(returns):
    """Calculate annualized volatility"""
    return returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(returns, risk_free_rate):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate):
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < risk_free_rate]
    
    if len(downside_returns) == 0:
        return np.nan
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns.mean() * 252 / downside_deviation

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(returns):
    """Calculate Calmar ratio"""
    annual_return = calculate_annualized_return(returns)
    max_dd = calculate_max_drawdown(returns)
    return annual_return / abs(max_dd)

def calculate_value_at_risk(returns, confidence=0.95):
    """Calculate Value at Risk"""
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_conditional_var(returns, confidence=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    var = calculate_value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate):
    """Calculate Jensen's alpha"""
    excess_portfolio = portfolio_returns - risk_free_rate
    excess_benchmark = benchmark_returns - risk_free_rate
    
    # Calculate beta first
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    
    # Calculate alpha
    alpha = excess_portfolio.mean() - beta * excess_benchmark.mean()
    return alpha * 252  # Annualize

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calculate beta to benchmark"""
    covariance = portfolio_returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    return covariance / benchmark_variance

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """Calculate tracking error"""
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.std() * np.sqrt(252)

def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculate information ratio"""
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.mean() / active_returns.std() * np.sqrt(252)

def calculate_up_capture(portfolio_returns, benchmark_returns):
    """Calculate up capture ratio"""
    up_market = benchmark_returns > 0
    if up_market.sum() == 0:
        return np.nan
    portfolio_up = portfolio_returns[up_market].mean()
    benchmark_up = benchmark_returns[up_market].mean()
    return portfolio_up / benchmark_up

def calculate_down_capture(portfolio_returns, benchmark_returns):
    """Calculate down capture ratio"""
    down_market = benchmark_returns < 0
    if down_market.sum() == 0:
        return np.nan
    portfolio_down = portfolio_returns[down_market].mean()
    benchmark_down = benchmark_returns[down_market].mean()
    return portfolio_down / benchmark_down

def calculate_win_rate(returns):
    """Calculate win rate (percentage of positive returns)"""
    return (returns > 0).mean()

def calculate_profit_factor(returns):
    """Calculate profit factor (gross profit / gross loss)"""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / gross_loss if gross_loss != 0 else np.inf

def align_series(series1, series2):
    """Align two time series"""
    aligned_index = series1.index.intersection(series2.index)
    return series1.loc[aligned_index], series2.loc[aligned_index]
