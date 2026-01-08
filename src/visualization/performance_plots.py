"""
Visualization module for performance and risk analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Creates comprehensive performance visualizations"""
    
    def __init__(self, config):
        self.config = config
        self.figsize = (16, 10)
        
    def create_all_plots(self, backtest_results, save_dir='results/plots'):
        """Create all performance plots"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Equity curve comparison
        self.plot_equity_curve(backtest_results, save_dir)
        
        # 2. Rolling Sharpe ratio
        self.plot_rolling_sharpe(backtest_results, save_dir)
        
        # 3. Drawdown chart
        self.plot_drawdown(backtest_results, save_dir)
        
        # 4. Monthly returns heatmap
        self.plot_monthly_returns_heatmap(backtest_results, save_dir)
        
        # 5. Risk-return scatter
        self.plot_risk_return_scatter(backtest_results, save_dir)
        
        # 6. Portfolio composition
        self.plot_portfolio_composition(backtest_results, save_dir)
        
        # 7. Correlation matrix
        self.plot_correlation_matrix(backtest_results, save_dir)
        
        # 8. Performance attribution
        self.plot_performance_attribution(backtest_results, save_dir)
        
    def plot_equity_curve(self, results, save_dir):
        """Plot equity curve vs benchmark"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + results['equity_curve']).cumprod()
        
        # Plot equity curve
        ax1.plot(portfolio_cumulative.index, portfolio_cumrative.values, 
                label='Portfolio', linewidth=2)
        
        # Plot benchmark if available
        if 'benchmark_returns' in results:
            benchmark_cumulative = (1 + results['benchmark_returns']).cumprod()
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values,
                    label='Benchmark', linewidth=2, alpha=0.7)
        
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot rolling returns (12-month)
        rolling_return = portfolio_cumulative.pct_change(252)
        ax2.plot(rolling_return.index, rolling_return.values, 
                color='green', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('12-Month Rolling Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rolling_sharpe(self, results, save_dir):
        """Plot rolling Sharpe ratio"""
        returns = results['equity_curve']
        
        # Calculate rolling Sharpe (6-month window)
        window = 126  # 6 months
        rolling_sharpe = returns.rolling(window).mean() / \
                        returns.rolling(window).std() * np.sqrt(252)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
                color='purple', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=rolling_sharpe.mean(), color='g', 
                  linestyle='--', alpha=0.7, label=f'Mean: {rolling_sharpe.mean():.2f}')
        
        ax.set_title('6-Month Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rolling_sharpe.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown(self, results, save_dir):
        """Plot drawdown chart"""
        returns = results['equity_curve']
        cumulative = (1 + returns).cumprod()
        
        # Calculate drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot equity curve with drawdown areas
        ax1.plot(cumulative.index, cumulative.values, 
                label='Portfolio', linewidth=2)
        ax1.fill_between(cumulative.index, cumulative.values, 
                        running_max.values, 
                        where=cumulative.values < running_max.values,
                        color='red', alpha=0.3, label='Drawdown')
        ax1.set_title('Equity Curve with Drawdown Periods', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown percentage
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        where=drawdown.values < 0,
                        color='red', alpha=0.5)
        ax2.plot(drawdown.index, drawdown.values, 
                color='darkred', linewidth=1)
        ax2.set_title('Drawdown Percentage', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown %', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Annotate max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.annotate(f'Max DD: {max_dd*100:.1f}%', 
                    xy=(max_dd_date, max_dd),
                    xytext=(max_dd_date, max_dd*0.5),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/drawdown_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_returns_heatmap(self, results, save_dir):
        """Plot monthly returns heatmap"""
        returns = results['equity_curve']
        
        # Create monthly returns dataframe
        monthly_returns = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Pivot for heatmap
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_table = monthly_returns_df.pivot(index='Year', 
                                              columns='Month', 
                                              values='Return')
        
        # Month names for x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
        
        # Add text annotations
        for i in range(len(pivot_table)):
            for j in range(len(pivot_table.columns)):
                text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.1%}',
                              ha='center', va='center', 
                              color='black', fontsize=8)
        
        ax.set_xticks(range(len(month_names)))
        ax.set_xticklabels(month_names)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        
        ax.set_title('Monthly Returns Heatmap (%)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/monthly_returns_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_risk_return_scatter(self, results, save_dir):
        """Plot risk-return scatter plot for all assets"""
        if 'asset_returns' not in results:
            return
        
        asset_returns = results['asset_returns']
        
        # Calculate annualized metrics for each asset
        annual_return = asset_returns.mean() * 252
        annual_vol = asset_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        scatter = ax.scatter(annual_vol, annual_return, 
                           c=sharpe, s=100, cmap='viridis', 
                           alpha=0.7, edgecolors='black')
        
        # Add labels for each point
        for i, symbol in enumerate(annual_return.index):
            ax.annotate(symbol, (annual_vol.iloc[i], annual_return.iloc[i]),
                       fontsize=8, alpha=0.7)
        
        # Add efficient frontier (simplified)
        x = np.linspace(annual_vol.min(), annual_vol.max(), 100)
        max_sharpe = sharpe.max()
        y = max_sharpe * x
        
        ax.plot(x, y, 'r--', alpha=0.5, label=f'Max Sharpe: {max_sharpe:.2f}')
        
        ax.set_title('Risk-Return Scatter Plot', fontsize=14, fontweight='bold')
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/risk_return_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_portfolio_composition(self, results, save_dir):
        """Plot portfolio composition over time"""
        weights = results['weights_history']
        
        # Top 10 holdings by average weight
        avg_weights = weights.mean().sort_values(ascending=False)
        top_symbols = avg_weights.head(10).index
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Top holdings over time
        top_weights = weights[top_symbols]
        axes[0].stackplot(top_weights.index, top_weights.T.values,
                         labels=top_symbols, alpha=0.8)
        axes[0].set_title('Top 10 Holdings Over Time', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Weight', fontsize=12)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Current allocation pie chart
        current_weights = weights.iloc[-1].sort_values(ascending=False)
        top_current = current_weights.head(8)
        other = current_weights[8:].sum()
        
        if other > 0:
            top_current['Other'] = other
        
        axes[1].pie(top_current.values, labels=top_current.index,
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Current Portfolio Allocation', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/portfolio_composition.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self, results, save_dir):
        """Plot correlation matrix of asset returns"""
        if 'asset_returns' not in results:
            return
        
        returns = results['asset_returns']
        correlation = returns.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation, mask=mask, cmap=cmap, 
                   center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_attribution(self, results, save_dir):
        """Plot performance attribution by factor/sector"""
        # This would require factor/sector data
        # Simplified version for demonstration
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Mock data - in practice, calculate actual attribution
        categories = ['Momentum', 'Value', 'Size', 'Quality', 'Volatility']
        attribution = [0.35, 0.25, 0.15, 0.15, 0.10]
        
        bars = ax.bar(categories, attribution)
        
        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_title('Performance Attribution by Factor', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Contribution to Returns', fontsize=12)
        ax.set_ylim(0, 0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, attribution):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1%}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_attribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
