🚀 Bitcoin Price Prediction Using LSTM

Author: Alec Stovari, 2025

📌 Overview

This project implements a Bitcoin price prediction model using an LSTM (Long Short-Term Memory) neural network. It leverages historical hourly price data from CoinGecko's API, extracts relevant features, and forecasts future prices with uncertainty bounds.

🔧 Features:


✅ Fetches Bitcoin price data (up to 90 days) from CoinGecko

✅ Feature engineering: Adds price percentage change, rolling standard deviation, and price difference

✅ Data normalization & sequence creation for time series forecasting

✅ LSTM model with two layers and dropout for better generalization

✅ Train-Test split for evaluating model performance

✅ Forecasts future prices with a retrained model using the entire dataset

✅ Uncertainty estimation using standard deviation of prediction errors

✅ Two plots:

Train & Test vs. Predictions , 
Train + Test + Forecasted Prices

🏗️ Future Improvements

- Add hyperparameter tuning for better accuracy
- Support other cryptocurrencies (Ethereum, Litecoin, etc.)
- Implement alternative models (GRU, Transformer-based models)
- Enhance uncertainty quantification using Bayesian deep learning

📜 License

This code may not be used by anyone not in possession of the <b>LSTM for Market Forecasting: A Python deep learning guide</b> book : 

https://www.amazon.com/dp/B0CDVFPH6X
