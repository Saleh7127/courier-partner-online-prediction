# Objective
The main objective of the model is to predict CPO (Courier Partners Online) for the next 7 days based on past weather data and CPO history. The model takes into account weather-related features (temperature, humidity, precipitation) along with historical data to predict the next CPO value.

# Steps
## 1. Data Preprocessing

### Rolling Averages
We calculated rolling averages over the last 7 days for the following features to create exogenous variables that could be used for predictions:
- **Temperature**: Rolling average of the temperature over the last 7 days.
- **Relative Humidity**: Rolling average of relative humidity over the last 7 days.
- **Precipitation**: Rolling average of precipitation levels over the last 7 days.

These rolling averages help smooth out short-term fluctuations and highlight longer-term trends, which are useful for prediction tasks.

### Feature Engineering
Additional features were created based on the timestamp data to capture temporal patterns and help improve the predictive power of the model:
- **dayofweek**: Day of the week (0 = Monday, 6 = Sunday).
- **quarter**: Quarter of the year (1, 2, 3, 4).
- **month**: Month of the year (1-12).
- **year**: Year (integer).
- **dayofyear**: Day of the year (1-365/366).
- **dayofmonth**: Day of the month (1-31).
- **weekofyear**: Week number of the year (1-52).
  
These additional features help the model understand seasonality, weekly patterns, and other temporal effects that can influence the prediction of the **Courier Partners Online (CPO)** variable.

## 2. Modeling
We used LightGBM, a gradient boosting framework, for the regression task. LightGBM is a powerful algorithm known for its efficiency and scalability.

## 3. Hyperparameter Optimization with Optuna
To fine-tune the LightGBM model, Optuna was used for hyperparameter optimization. Optuna is an automatic hyperparameter optimization framework that efficiently searches for the best combination of hyperparameters by using algorithms such as tree-structured Parzen estimators (TPE).

## 4. Prediction
The final model predicts the CPO for the next 7 days, using the historical data along with weather-related exogenous features.

## 5. Optimization Process
Using Optuna, we search for the optimal hyperparameters with 50 trials.

# Results
After optimizing the hyperparameters, the model provided the predicted CPO for the next 7 days. The optimized model has improved performance compared to the base LightGBM model.

## Dependencies

- Python 3.x
- Pandas
- Numpy
- LightGBM
- Optuna

## Installation

To install the required dependencies, you can use the following pip command:

```bash
pip install pandas numpy lightgbm optuna

