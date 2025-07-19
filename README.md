# 📈 Quantitative Research and Portfolio Optimization

**Timeframe**: Nov 2023 – Jan 2024
**Language**: Python
**Libraries**: `pandas`, `numpy`, `yfinance`, `matplotlib`, `scikit-learn`, `statsmodels`, `seaborn`

---

## 🚀 Overview

This project involves the quantitative design and analysis of multi-asset portfolios using Python. We focus on building systematic investment strategies through predictive modeling, risk-adjusted portfolio construction, and robust backtesting techniques.

The goal was to outperform standard benchmarks by applying modern financial theory, statistical modeling, and data science practices to real-world market data.

---

## 🧠 Key Features and Methodologies

### 📊 1. **Custom Alpha Signal Generation**

* Constructed alpha signals using:

  * Historical return momentum (weekly/monthly).
  * Beta sensitivity constraints.
  * Cross-sectional volatility filters.
* Designed to exploit inefficiencies in sector movements and volatility clustering.

### 🧮 2. **Portfolio Construction Techniques**

* Built multiple portfolio types:

  * **Sharpe-optimized portfolios** using mean-variance optimization.
  * **Linear regression-based exposure models** to determine ideal asset weights.
  * **Covariance-adjusted allocation schemes** to better handle correlated assets.
* Enforced realistic constraints including weight caps and beta neutrality.

### 🧪 3. **Backtesting and Validation**

* Implemented walk-forward validation:

  * Rolling time windows for training/testing over 3-year and 5-year periods.
  * Included realistic constraints like:

    * **Transaction costs**
    * **Sector-neutral rebalancing**
    * **Position limits**
* Evaluated using key metrics:

  * Annualized Sharpe & Treynor Ratios
  * Jensen’s Alpha
  * Max Drawdown and Volatility

### 📉 4. **Risk and Performance Comparison**

* Compared portfolio vs individual stock risk-reward metrics.
* Highlighted diversification benefits using correlation and volatility decomposition.
* Constructed **correlation-based alternative portfolios** to reduce exposure clustering.

---

## 📈 Results

* **+17% Alpha** over benchmark (S\&P 500 and NIFTY 50 equivalents).
* **22% Lower portfolio risk** (measured by annualized standard deviation).
* Outperformance was consistent across multiple market regimes (growth, volatility, correction).

---

Here's the full **code structure** for your **Quantitative Research and Portfolio Optimization** project, based on the improved README.

---

### 📁 Project Directory Structure

```
quant-portfolio-optimization/
│
├── data/                           # Raw data and processed CSVs
│   ├── sp500_prices.csv
│   ├── nifty50_prices.csv
│   └── sector_info.csv
│
├── notebooks/                      # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_alpha_signal_generation.ipynb
│   ├── 03_portfolio_construction.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_performance_analysis.ipynb
│
├── models/                         # Custom strategy and portfolio models
│   ├── signal_generator.py
│   ├── optimizer.py
│   ├── backtester.py
│   └── risk_metrics.py
│
├── utils/                          # Helper functions and constants
│   ├── data_loader.py
│   ├── finance_utils.py
│   ├── regression_tools.py
│   └── plot_utils.py
│
├── results/                        # Saved results and plots
│   ├── performance_summary.csv
│   ├── sharpe_plot.png
│   └── correlation_heatmap.png
│
├── main.py                         # End-to-end script (loads data → runs full pipeline)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

---

### 📝 File Descriptions

#### `main.py`

* Loads stock/ETF data.
* Runs alpha signal generator.
* Constructs optimized portfolios.
* Performs backtesting and exports results.

#### `models/`

* `signal_generator.py`: Computes momentum, beta constraints, and volatility signals.
* `optimizer.py`: Sharpe maximization, regression-based weight setting, etc.
* `backtester.py`: Rolling-window backtest engine with transaction cost modeling.
* `risk_metrics.py`: Calculates Sharpe, Alpha, Beta, Drawdown, etc.

#### `utils/`

* `data_loader.py`: Cleans, resamples, and aligns stock data.
* `finance_utils.py`: CAPM beta, rolling returns, risk contributions.
* `regression_tools.py`: Linear regression for factor exposure.
* `plot_utils.py`: Returns curve, heatmaps, volatility charts.

#### `notebooks/`

* Serve as EDA, experimentation, and visualization tools.
* Each step of the pipeline is modularized for quick validation.

---

### ✅ requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
yfinance
scipy
```

---

## 📌 Dependencies

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

---

## 📊 Example Visualizations

* Return Heatmaps
* Risk Contribution by Asset
* Walk-Forward Equity Curve
* Rolling Sharpe & Beta Plots
* Correlation Matrix for Diversification Planning

---

## 💡 Learnings and Takeaways

* Feature engineering (alpha signals) greatly improves portfolio returns when tuned carefully.
* Sector and correlation-aware rebalancing reduces drawdowns significantly.
* Backtest realism (transaction cost, slippage) is crucial to avoid overfitting.
* Statistical modeling combined with market intuition gives the best results in portfolio optimization.

---

## 📎 Future Improvements

* Integrate LSTM or transformer-based models for alpha signal prediction.
* Automate ETF selection for dynamic universe generation.
* Deploy as a streamlit dashboard for interactive performance visualization.

 
