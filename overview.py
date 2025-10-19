import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data for two stocks
tickers = ["AAPL", "MSFT"]
data = yf.download(tickers, start="2023-01-01", end="2025-10-18")['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate portfolio returns assuming equal weights
weights = np.array([0.5, 0.5])
portfolio_returns = returns.dot(weights)

# Calculate 95% Historical VaR and Expected Shortfall (ES)
confidence_level = 0.05
VaR_95 = np.percentile(portfolio_returns, confidence_level*100)
print(f"95% Historical VaR: {VaR_95:.2%}")

ES_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()
print(f"95% Expected Shortfall: {ES_95:.2%}")

# Plot the distribution of portfolio returns
plt.hist(portfolio_returns, bins=50, color='skyblue', edgecolor='black')
plt.axvline(VaR_95, color='red', linestyle='dashed', linewidth=2, label='VaR 95%')
plt.axvline(ES_95, color='orange', linestyle='dashed', linewidth=2, label='ES 95%')
plt.title("Portfolio Daily Returns Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()