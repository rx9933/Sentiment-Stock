finance-models: 
- includes models for random forest, lightgbm, xgboost, lstm-gru, lstm. In order of decreasing accuracy.
- start_*: 15 day standard input stencil -> 15 day standard output prediction. * indicates index in testing data.
- future.png: future predictions (11_25 -> 11_25 + 9 business day) are included. 9 days in --> 9 days out. 9 is used instead of 15 due to the sharp rise in stock ~ 14 days ago (250->350 stock price (Musk/Ramaswamy DOGE??) in 3 days) causing instability in predictions.
- recheck_25.png: using future prediction code to calculate start_25 prediction for rechecking purposes.
- log_volatility_vs_mse.png: measure of error and stock volatility.

### Logarithmic Returns Volatility

Logarithmic returns are commonly used in finance to measure volatility. Volatility, in this context, is the standard deviation of log returns, which captures proportional changes and is less biased by outliers. This is especially useful when working with time-series models.

#### Log Return:
The log return for a given time period is calculated as:

\[
\text{Log Return} = \ln\left( \frac{y[i]}{y[i-1]} \right)
\]

Where:
- \( y[i] \) is the price or value at time \( i \)
- \( y[i-1] \) is the price or value at the previous time \( i-1 \)

#### Volatility:
Volatility is the standard deviation of the log returns over a specific period:

\[
\text{Volatility} = \text{std}(\text{Log Returns})
\]

This measure captures the variability in the returns over time.
############
yfinance:
- get_data.ipynb: to generate new sentiment data. 

