# Stock Price Prediction

## How to Run
  `python3 stock_price_prediction.py`
  
  Then type in the ticker symbol (e.g. AAPL) you want to predict.
  
## Methods
We use Yahoo Finance API to fetch stock prices.

## Model
### LSTM (Long Short Term Memory)
We use past 60 days to predict one day ahead, and train the model with 25 epochs.

## Result
The following image shows the difference between the actual stock prices and the predicted stock prices for validation data, and the prediction of tomorrow stock price is shown at the end.

![image](https://github.com/wutonytt/Stock-Price-Prediction/blob/0675b3d72485f7e65f9c4bcec97cffef5113d206/Figure_1.png)

Prediction: 134.3729
