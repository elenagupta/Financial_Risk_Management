import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Portfolio setup
tickers = ["AAPL", "MSFT", "NVDA"]
weights = np.array([0.4, 0.4, 0.2])  # must sum to 1
start_date = "2023-01-01"
end_date = "2025-10-18"

data = yf.download(tickers, start=start_date, end=end_date)['Close']
returns = data.pct_change().dropna()

portfolio_returns = returns.dot(weights)

# Historical VaR and ES
confidence = 0.05
VaR_hist = np.percentile(portfolio_returns, confidence*100)
ES_hist = portfolio_returns[portfolio_returns <= VaR_hist].mean()

mean_ret = portfolio_returns.mean()
std_ret = portfolio_returns.std()

# z-score for 95% confidence (5% left tail)
z = -1.645
VaR_param = mean_ret + z * std_ret
ES_param = mean_ret - (std_ret * (np.exp(-z**2/2) / (np.sqrt(2*np.pi)*0.05)))

# Monte Carlo Simulation
mean_vec = returns.mean().values
cov_matrix = returns.cov().values

n_sims = 10000
simulated_returns = np.random.multivariate_normal(mean_vec, cov_matrix, n_sims)
sim_portfolio = simulated_returns.dot(weights)

VaR_mc = np.percentile(sim_portfolio, confidence*100)
ES_mc = sim_portfolio[sim_portfolio <= VaR_mc].mean()

# Output results
portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
marginal_contrib = np.dot(cov_matrix, weights)
risk_contrib = weights * marginal_contrib / portfolio_var

risk_table = pd.DataFrame({
    "Ticker": tickers,
    "Weight": weights,
    "Risk Contribution (%)": risk_contrib * 100
})
print(risk_table)

# Summary of VaR and ES
VaR_summary = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "VaR (95%)": [VaR_hist, VaR_param, VaR_mc],
    "ES (95%)": [ES_hist, ES_param, ES_mc]
})

print(VaR_summary)

# Plotting the distribution of portfolio returns
plt.figure(figsize=(8,5))
plt.hist(portfolio_returns, bins=50, color='skyblue', edgecolor='black')
plt.axvline(VaR_hist, color='red', linestyle='dashed', linewidth=2, label='Historical VaR')
plt.axvline(ES_hist, color='orange', linestyle='dashed', linewidth=2, label='Historical ES')
plt.title("Portfolio Daily Returns Distribution")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Putting them into one file
with open("risk_report.txt", "w") as f:
    f.write("Risk Contribution Table:\n")
    f.write(risk_table.to_string(index=False))
    f.write("\n\nVaR and ES Summary:\n")
    f.write(VaR_summary.to_string(index=False))