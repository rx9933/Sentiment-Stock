# Finance Models

This repository includes various models for stock price prediction, ranked in order of decreasing accuracy:

- **Random Forest**
- **LightGBM**
- **XGBoost**
- **LSTM-GRU**
- **LSTM**

### Model Details
- **start_**: A 15-day standard input stencil is used, where the model predicts stock prices for the next 15 days based on the previous 15 days of data. The `*` in `start_*` indicates the index in the testing data.
- **future.png**: Contains future predictions for the stock price. The predictions cover a period from **11/25** to **11/25 + 9 business days** (9 days in and 9 days out). The number 9 is used instead of 15 days due to a sharp rise in stock prices around **14 days ago** (from **$250** to **$350**, potentially due to Musk/Ramaswamy DOGE-related activity). This sharp increase caused instability in the predictions, hence a shorter prediction window is used.
- **recheck_25.png**: Contains predictions for the `start_25` index, which are used to recheck the model’s predictions for validation purposes.
- **log_volatility_vs_mse.png**: This plot measures the relationship between error and stock volatility.

### Logarithmic Returns Volatility

Logarithmic returns are widely used in finance to measure volatility. In this context, **volatility** is calculated as the standard deviation of the log returns. Logarithmic returns capture proportional changes and are less biased by outliers, making them ideal for time-series analysis.

#### Log Return:
The log return for a given time period is calculated using the following formula:

$$
\text{Log Return} = \ln\left( \frac{y[i]}{y[i-1]} \right)
$$

Where:
- \( y[i] \) is the price or value at time \( i \)
- \( y[i-1] \) is the price or value at the previous time \( i-1 \)

#### Volatility:
Volatility is the standard deviation of the log returns over a specific period:

\[
\text{Volatility} = \text{std}(\text{Log Returns})
\]

This metric captures the variability in the stock returns over time.

---

### yFinance

- **get_data.ipynb**: generate new sentiment data by extracting stock information via the Yahoo Finance API. 
