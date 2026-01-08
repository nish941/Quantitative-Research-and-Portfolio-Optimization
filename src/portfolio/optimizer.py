"""
Portfolio optimization module
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from loguru import logger

class PortfolioOptimizer:
    """Implements various portfolio optimization techniques"""
    
    def __init__(self, config):
        self.config = config
        
    def optimize_portfolio(self, signals, price_data):
        """Main optimization method"""
        logger.info("Optimizing portfolio...")
        
        # Calculate expected returns from signals
        expected_returns = self.calculate_expected_returns(signals, price_data)
        
        # Calculate covariance matrix
        returns = price_data.pct_change().dropna()
        cov_matrix = self.calculate_covariance_matrix(returns)
        
        # Get different portfolio allocations
        portfolios = {}
        
        # 1. Mean-Variance Optimization (Markowitz)
        portfolios['markowitz'] = self.mean_variance_optimization(
            expected_returns, cov_matrix
        )
        
        # 2. Risk Parity
        portfolios['risk_parity'] = self.risk_parity_allocation(cov_matrix)
        
        # 3. Minimum Variance
        portfolios['min_variance'] = self.minimum_variance_portfolio(cov_matrix)
        
        # 4. Maximum Sharpe Ratio
        portfolios['max_sharpe'] = self.maximum_sharpe_portfolio(
            expected_returns, cov_matrix
        )
        
        # 5. Hierarchical Risk Parity (simplified)
        portfolios['hrp'] = self.hierarchical_risk_parity(cov_matrix)
        
        # Combine strategies (ensemble approach)
        final_weights = self.combine_portfolios(portfolios)
        
        # Apply constraints
        final_weights = self.apply_constraints(final_weights)
        
        logger.info(f"Optimized portfolio with {len(final_weights)} assets")
        return final_weights
    
    def calculate_expected_returns(self, signals, price_data):
        """Calculate expected returns from alpha signals"""
        # Convert signals to expected returns
        # Simple approach: signal strength correlates with expected returns
        returns = price_data.pct_change().dropna()
        
        # Align signals with returns
        aligned_signals = signals.reindex(returns.index).ffill()
        
        # Scale signals to expected returns
        # Assuming 15% annual target volatility
        target_vol = self.config.TARGET_VOLATILITY / np.sqrt(252)
        
        # Normalize signals
        normalized_signals = (
            aligned_signals - aligned_signals.mean()
        ) / aligned_signals.std()
        
        # Convert to expected daily returns
        expected_returns = normalized_signals * target_vol * 0.1  # Scale factor
        
        return expected_returns
    
    def calculate_covariance_matrix(self, returns):
        """Calculate covariance matrix with shrinkage"""
        # Basic covariance
        cov = returns.cov() * 252  # Annualize
        
        # Apply shrinkage towards diagonal
        shrinkage = 0.5
        n_assets = returns.shape[1]
        F = np.diag(np.diag(cov))  # Diagonal matrix
        cov_shrunk = shrinkage * F + (1 - shrinkage) * cov
        
        # Ensure positive semi-definite
        cov_shrunk = self.make_psd(cov_shrunk)
        
        return cov_shrunk
    
    def make_psd(self, matrix, epsilon=1e-6):
        """Make matrix positive semi-definite"""
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Replace negative eigenvalues with epsilon
        eigvals = np.maximum(eigvals, epsilon)
        
        # Reconstruct matrix
        psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return psd_matrix
    
    def mean_variance_optimization(self, expected_returns, cov_matrix):
        """Mean-Variance Optimization using convex optimization"""
        n_assets = len(expected_returns.columns)
        
        # Define variables
        weights = cp.Variable(n_assets)
        
        # Define objective: maximize return - risk_aversion * variance
        risk_aversion = 1.0
        portfolio_return = expected_returns.iloc[-1].values @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= self.config.MIN_WEIGHT,  # Minimum weight
            weights <= self.config.MAX_WEIGHT,  # Maximum weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if weights.value is not None:
                return pd.Series(weights.value, index=expected_returns.columns)
        except:
            logger.warning("CVXPY optimization failed, using equal weights")
        
        return pd.Series(1/n_assets, index=expected_returns.columns)
    
    def risk_parity_allocation(self, cov_matrix):
        """Risk Parity portfolio allocation"""
        # Simplified risk parity: inverse volatility weighting
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return pd.Series(weights, index=cov_matrix.index)
    
    def minimum_variance_portfolio(self, cov_matrix):
        """Minimum Variance portfolio"""
        n_assets = len(cov_matrix)
        
        def objective(weights):
            return weights @ cov_matrix.values @ weights
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, x0, 
                         bounds=bounds, 
                         constraints=constraints)
        
        return pd.Series(result.x, index=cov_matrix.index)
    
    def maximum_sharpe_portfolio(self, expected_returns, cov_matrix):
        """Maximum Sharpe Ratio portfolio"""
        n_assets = len(expected_returns.columns)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        def negative_sharpe(weights):
            portfolio_return = expected_returns.iloc[-1].values @ weights
            portfolio_variance = weights @ cov_matrix.values @ weights
            sharpe = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(negative_sharpe, x0, 
                         bounds=bounds, 
                         constraints=constraints)
        
        return pd.Series(result.x, index=expected_returns.columns)
    
    def hierarchical_risk_parity(self, cov_matrix):
        """Simplified Hierarchical Risk Parity"""
        # Compute correlation matrix
        corr_matrix = self.cov_to_corr(cov_matrix)
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0)
        
        # Perform hierarchical clustering
        condensed_dist = squareform(dist_matrix)
        Z = linkage(condensed_dist, method='ward')
        
        # Get clusters
        clusters = fcluster(Z, t=0.5, criterion='distance')
        
        # Allocate risk equally across clusters, then within clusters
        unique_clusters = np.unique(clusters)
        weights = np.zeros(len(cov_matrix))
        
        for cluster in unique_clusters:
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
            
            # Inverse volatility weighting within cluster
            cluster_vol = np.sqrt(np.diag(cluster_cov))
            cluster_weights = (1 / cluster_vol)
            cluster_weights = cluster_weights / cluster_weights.sum()
            
            # Distribute equally across clusters
            weights[cluster_indices] = cluster_weights / len(unique_clusters)
        
        return pd.Series(weights, index=cov_matrix.index)
    
    def cov_to_corr(self, cov):
        """Convert covariance matrix to correlation matrix"""
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        return corr
    
    def combine_portfolios(self, portfolios):
        """Combine multiple portfolio strategies"""
        # Simple average of all strategies
        combined = pd.concat(portfolios.values(), axis=1).mean(axis=1)
        
        # Ensure weights sum to 1
        combined = combined / combined.sum()
        
        return combined
    
    def apply_constraints(self, weights):
        """Apply portfolio constraints"""
        # 1. Sector neutrality (simplified)
        # In practice, you would map symbols to sectors
        
        # 2. Turnover constraint
        # Implemented during rebalancing
        
        # 3. Beta constraint
        # Would require benchmark data
        
        # 4. Long-only constraint (already applied)
        weights = weights.clip(lower=self.config.MIN_WEIGHT, 
                              upper=self.config.MAX_WEIGHT)
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
