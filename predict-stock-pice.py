import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytz
import yfinance as yf

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# SQL Alchemy
from sqlalchemy import create_engine
# from config import username, password

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

start = "2010-01-01"

###
# Apple Inc. (AAPL)
# Alphabet Inc. (GOOGL)
# Microsoft Corporation (MSFT)
# Amazon.com, Inc. (AMZN)
# Meta (formerly Facebook) Inc. (META)
# Tesla Motors (TSLA)
# The Goldman Sachs Group, Inc. (GS)
# The Dow Jones Industrial Average (DJIA)
# The S&P 500 Index (SPX)
# The NASDAQ Composite Index (COMP)
###

ticker = "AMZN"
data_DF = yf.download(ticker, start=start, period="ytd")


data_DF.head(5)

data_DF = data_DF.reset_index(level=0) #reset indexes
data_DF = data_DF.dropna() #delete rows with null values
df_ticker = data_DF.drop_duplicates() #delete duplicate rows

# Rename the columns
df_ticker = df_ticker.rename(columns={ #rename columns
    'Date': 'Date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adjclose',
    'Volume': 'volume'
})

df_ticker.to_csv(f'content/{ticker}.csv', sep=',', index=True) #load data to CSV-file

# Create Engine for project data
conn = create_engine("sqlite:///content/stocks.db") #connect to SQLite DB
conn2 = create_engine("sqlite:///content/stocks.db")
connection = conn2.raw_connection()
df_ticker.to_sql(ticker, con=conn, if_exists='replace', index=False) #load data to Database

df_ticker = df_ticker.set_index('Date')

#linear regression

#Keep only the close data
df = df_ticker[['close']]
df = df.dropna()

df['S_3'] = df['close'].shift(1).rolling(window=3).mean()
df['S_9'] = df['close'].shift(1).rolling(window=9).mean()
df = df.dropna()
X = df[['S_3', 'S_9']]
X.head()

y = df['close']
y.head()

#Split the data into train and test dataset
t = .8
t = int(t * len(df))
# Train dataset
X_train = X[:t]
y_train = y[:t]
# Test dataset
X_test = X[t:]
y_test = y[t:]

#Create a linear regression model and fit the data
linear = LinearRegression().fit(X_train, y_train)

#Prediction using test data
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(12, 6))
y_test.plot()
title_reg = ticker + ' Linear Regression Prediction'
plt.title(title_reg)
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Price")
plt.show()

#show the scores for test and train data predictions
print("R2 score for linear regression model : %.2f" % r2_score(y_test, predicted_price))

df = df.reset_index()

useast = datetime.now(pytz.timezone('America/New_York'))
useast = useast.strftime('%Y-%m-%d')
useast = datetime.strptime(useast, '%Y-%m-%d')
first_forecast_date = useast + timedelta(1)

#loop to add seven rows of dummy data
for i in range(7):
    df.loc[len(df)] = df.loc[len(df)-1]
    next_day = useast + timedelta(i+1)
    df.iloc[-1, df.columns.get_loc('Date')] = next_day
    df['Date'] = pd.to_datetime(df["Date"], utc=True).dt.date

df = df.set_index('Date')

x_today = df.iloc[-7:]
x_today = x_today[['S_3', 'S_9']]

#predict tomorrow's price
new_price = linear.predict(x_today)

reg_forecast = pd.DataFrame(new_price, index=x_today.index, columns=['forecast'])
reg_forecast.to_csv(f'content/{ticker}_reg_forecast.csv', sep=',', index=True)

cur = connection.cursor()
cur.execute(f'CREATE TABLE {ticker}_forecast_linear (date, forecast);') # create a new table

with open(f'content/{ticker}_reg_forecast.csv', 'r') as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['Date'], i['forecast']) for i in dr]

cur.executemany(f'INSERT INTO {ticker}_forecast_linear (date, forecast) VALUES (?, ?);', to_db)
connection.commit()

#Prediction using test data
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(12, 6))
y_test.plot()
title_reg = ticker + ' Linear Regression Prediction'
plt.title(title_reg)
plt.plot(reg_forecast['forecast'], color='Yellow')
plt.ylabel("Price")
plt.legend(['Test', 'Predictions', 'Forecast'], loc='lower left')
plt.show()



# #############################################
# #               LSTM MODEL 1                #
# #############################################


df_ticker = df_ticker.reset_index(level=0)

df_ticker.head(5)
#
## Add a dummy row at the end. This will not be used to predict.
useast = datetime.now(pytz.timezone('America/New_York'))
useast = useast.strftime('%Y-%m-%d')
useast = datetime.strptime(useast, '%Y-%m-%d')
first_forecast_date = useast + timedelta(1)

#loop to add seven rows of dummy data
for i in range(7):
    df_ticker.loc[len(df_ticker)]=df_ticker.loc[len(df_ticker)-1]
    next_day = useast + timedelta(i+1)
    df_ticker.iloc[-1, df_ticker.columns.get_loc('Date')] = next_day
    df_ticker['Date'] = pd.to_datetime(df_ticker["Date"], utc=True).dt.date

df_ticker = df_ticker.set_index('Date')

#Set Target Variable
output_var = pd.DataFrame(df_ticker["adjclose"], index=df_ticker.index)
#Selecting the Features
features = ['open', 'high', 'low', 'volume']

#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df_ticker[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df_ticker.index)
feature_transform.tail()

timesplit = TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)], output_var[len(train_index): (len(train_index)+len(test_index))]

#Process the data for LSTM
trainX = np.array(X_train)
testX = np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])


#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the Training set
history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred = lstm.predict(X_test)

#Predicted vs True Adj Close Value – LSTM
predicted_df = pd.DataFrame(y_test)
predicted_df['predictions'] = y_pred

forecat_lstm_1 = predicted_df.tail(7)
forecat_lstm_1.to_csv(f'content/{ticker}_lstm_1_forecast.csv', sep=',', index=True)
cur.execute(f'CREATE TABLE {ticker}_forecast_lstm_1 (date, forecast);') # use your column names here

with open(f'content/{ticker}_lstm_1_forecast.csv', 'r') as fin: # `with` statement available in 2.5+
    # csv.DictReader uses first line in file for column headings by default
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['Date'], i['predictions']) for i in dr]

cur.executemany(f'INSERT INTO {ticker}_forecast_lstm_1 (date, forecast) VALUES (?, ?);', to_db)
connection.commit()

plt.figure(figsize=(12, 6))
plt.plot(predicted_df['adjclose'], label='True Value')
plt.plot(predicted_df['predictions'], label='LSTM Value')
title_name2 = ticker + ' Prediction by LSTM'
plt.title(title_name2)
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.savefig('LSTM_Predicted_Vs_AdjClose_Method_1.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(predicted_df['adjclose'], label='True Value')
plt.plot(predicted_df['predictions'], label='LSTM Value')
plt.plot(predicted_df['predictions'].tail(7), label='Forecast', color='Yellow')
title_name2 = ticker + ' Prediction by LSTM'
plt.title(title_name2)
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.savefig('LSTM_Predicted_Vs_AdjClose_Method_1.png')
plt.show()

forecast = predicted_df.tail(7)

#converting to equal format
y_test_adjclose = (y_test['adjclose']).to_numpy()
y_pred_rav = y_pred.ravel()

print("R2 score LSTM Method-1 : %.2f" % r2_score(y_test_adjclose, y_pred_rav))
#
lstm.save("content/models/LSTM_Method_1.h5")


#############################################
#               LSTM MODEL 2                #
#############################################

# Create a dataframe with only the Close Stock Price Column
data_target = df_ticker.filter(['close'])

# Convert the dataframe to a numpy array to train the LSTM model
target = data_target.values

# Splitting the dataset into training and test
# Target Variable: Close stock price value

training_data_len = int(len(target) * 0.75) # training set has 75% of the data
training_data_len

# Normalizing data before model fitting using MinMaxScaler
# Feature Scaling

sc = MinMaxScaler(feature_range=(0, 1))
training_scaled_data = sc.fit_transform(target)
training_scaled_data

# Create a training dataset containing the last 180-day closing price values we want to use to estimate the 181st closing price value.
train_data = training_scaled_data[0:training_data_len, :]

X_train = []
y_train = []
for i in range(180, len(train_data)):
    X_train.append(train_data[i-180:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train) # converting into numpy sequences to train the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #(1806 values, 180 time-steps, 1 output)

# We add the LSTM layer and later add a few Dropout layers to prevent overfitting.
# Building a LTSM model with 50 neurons and 4 hidden layers. We add the LSTM layer with the following arguments:
# 50 units which is the dimensionality of the output space
# return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence input_shape as the shape of our training set.
# When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped.
# Thereafter, we add the Dense layer that specifies the output of 1 unit.
# After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.

model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')
num_params = model.count_params()

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Getting the predicted stock price
test_data = training_scaled_data[training_data_len - 180:, :]

#Create the x_test and y_test data sets
X_test = []
y_test = target[training_data_len:, :]
for i in range(180, len(test_data)):
    X_test.append(test_data[i-180:i, 0])

# Convert x_test to a numpy array
X_test = np.array(X_test)

#Reshape the data into the shape accepted by the LSTM
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions using the test dataset
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
train = data_target[:training_data_len]
predicted_DF2 = data_target[training_data_len:]
predicted_DF2['predictions'] = predicted_stock_price
plt.figure(figsize=(12, 6))
title_name3 = ticker + ' Prediction by LSTM'
plt.title(title_name3)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(train['close'])
plt.plot(predicted_DF2[['close', 'predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='upper left')
plt.show()

predicted_DF2.tail(10)

predicted_DF2 = predicted_DF2.reset_index(level=0)

predicted_DF2.tail(10)

#filtering the forecasted stock price for the next 7 future days
filtered_df = predicted_DF2.loc[(predicted_DF2['Date'] >= first_forecast_date.date())]

filtered_df = filtered_df.set_index('Date')

filtered_df

filtered_df.to_csv(f'content/{ticker}_lstm_2_forecast.csv', sep=',', index=True)
cur.execute(f'CREATE TABLE {ticker}_forecast_lstm_2 (date, forecast);') # use your column names here

with open(f'content/{ticker}_lstm_2_forecast.csv', 'r') as fin:
    dr = csv.DictReader(fin)
    to_db = [(i['Date'], i['predictions']) for i in dr]

cur.executemany(f'INSERT INTO {ticker}_forecast_lstm_2 (date, forecast) VALUES (?, ?);', to_db)
connection.commit()

# Visualising the results
train = data_target[:training_data_len]
predicted_DF2 = data_target[training_data_len:]
predicted_DF2['predictions'] = predicted_stock_price
plt.figure(figsize=(10, 5))
title_name2 = ticker + ' Prediction by LSTM'
plt.title(title_name2)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Close Price USD ($)', fontsize=12)
plt.plot(predicted_DF2[['close', 'predictions']])
plt.plot(filtered_df['predictions'], color='Yellow')
plt.legend(['Test', 'Predictions', 'Forecast'], loc='lower left')
plt.show()

#R2 score
print("R2 score LSTM Method_2 : %.2f" % r2_score(y_test, predicted_stock_price))

#save models
model.save("content/models/LSTM_Method_2.h5")


