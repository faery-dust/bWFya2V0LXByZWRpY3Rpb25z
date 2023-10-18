import os
import requests
import dateutil.parser
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler


---Part of Code to be changed according to your own app's API and methods---

class KiteApp:

    # URIs to various calls
    _routes = {
        "user.profile": "/user/profile",
        "user.margins": "/user/margins",
        "user.margins.segment": "/user/margins/{segment}"}

    # Products
    PRODUCT_MIS = "MIS"
    PRODUCT_CNC = "CNC"
    PRODUCT_NRML = "NRML"
    PRODUCT_CO = "CO"

    # Order types
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SLM = "SL-M"
    ORDER_TYPE_SL = "SL"

    # Varities
    VARIETY_REGULAR = "regular"
    VARIETY_CO = "co"
    VARIETY_AMO = "amo"

    # Transaction type
    TRANSACTION_TYPE_BUY = "BUY"
    TRANSACTION_TYPE_SELL = "SELL"

    # Validity
    VALIDITY_DAY = "DAY"
    VALIDITY_IOC = "IOC"

    # Exchanges
    EXCHANGE_NSE = "NSE"
    EXCHANGE_BSE = "BSE"
    EXCHANGE_NFO = "NFO"
    EXCHANGE_CDS = "CDS"
    EXCHANGE_BFO = "BFO"
    EXCHANGE_MCX = "MCX"

    def __init__(self, enctoken):
        self.headers = {"Authorization": f"enctoken {enctoken}"}
        self.session = requests.session()
        self.root_url = "https://api.kite.trade"
        # self.root_url = "https://kite.zerodha.com/oms"
        self.session.get(self.root_url, headers=self.headers)

    def instruments(self, exchange=None):
        data = self.session.get(f"{self.root_url}/instruments",headers=self.headers).text.split("\n")
        Exchange = []
        for i in data[1:-1]:
            row = i.split(",")
            if exchange is None or exchange == row[11]:
                Exchange.append({'instrument_token': int(row[0]), 'exchange_token': row[1], 'tradingsymbol': row[2],
                                 'name': row[3][1:-1], 'last_price': float(row[4]),
                                 'expiry': dateutil.parser.parse(row[5]).date() if row[5] != "" else None,
                                 'strike': float(row[6]), 'tick_size': float(row[7]), 'lot_size': int(row[8]),
                                 'instrument_type': row[9], 'segment': row[10],
                                 'exchange': row[11]})
        return Exchange

    def quote(self, instruments):
        data = self.session.get(f"{self.root_url}/quote", params={"i": instruments}, headers=self.headers).json()["data"]
        return data

    def ltp(self, instruments):
        data = self.session.get(f"{self.root_url}/quote/ltp", params={"i": instruments}, headers=self.headers).json()["data"]
        return data

    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
        params = {"from": from_date,
                  "to": to_date,
                  "interval": interval,
                  "continuous": 1 if continuous else 0,
                  "oi": 1 if oi else 0}
        lst = self.session.get(
            f"{self.root_url}/instruments/historical/{instrument_token}/{interval}", params=params,
            headers=self.headers).json()["data"]["candles"]
        records = []
        for i in lst:
            record = {"date": dateutil.parser.parse(i[0]), "open": i[1], "high": i[2], "low": i[3],
                      "close": i[4], "volume": i[5],}
            if len(i) == 7:
                record["oi"] = i[6]
            records.append(record)
        return records

    def margins(self):
        margins = self.session.get(f"{self.root_url}/user/margins", headers=self.headers).json()["data"]
        return margins

    def orders(self):
        orders = self.session.get(f"{self.root_url}/orders", headers=self.headers).json()["data"]
        return orders

    def positions(self):
        positions = self.session.get(f"{self.root_url}/portfolio/positions", headers=self.headers).json()["data"]
        return positions

    def place_order(self, variety, exchange, tradingsymbol, transaction_type, quantity, product, order_type, price=None,
                    validity=None, disclosed_quantity=None, trigger_price=None, squareoff=None, stoploss=None,
                    trailing_stoploss=None, tag=None):
        params = locals()
        del params["self"]
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]
        order_id = self.session.post(f"{self.root_url}/orders/{variety}",
                                     data=params, headers=self.headers).json()["data"]["order_id"]
        return order_id

    def modify_order(self, variety, order_id, parent_order_id=None, quantity=None, price=None, order_type=None,
                     trigger_price=None, validity=None, disclosed_quantity=None):
        params = locals()
        del params["self"]
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        order_id = self.session.put(f"{self.root_url}/orders/{variety}/{order_id}",
                                    data=params, headers=self.headers).json()["data"][
            "order_id"]
        return order_id

    def cancel_order(self, variety, order_id, parent_order_id=None):
        order_id = self.session.delete(f"{self.root_url}/orders/{variety}/{order_id}",
                                       data={"parent_order_id": parent_order_id} if parent_order_id else {},
                                       headers=self.headers).json()["data"]["order_id"]
        return order_id
    def profile(self):
        """Get user profile details."""
        return self._get("user.profile")

"""# Fill in API auth token"""

enctoken = input()

kite = KiteApp(enctoken=enctoken)

all_instruments = kite.instruments("NSE")

data = pd.DataFrame(all_instruments,index=range(len(all_instruments)))

data.head()

"""## Search for viable instruments"""

print(len([s for s in data.tradingsymbol.unique() if s.isalpha()]))
named_stocks = [s for s in data.tradingsymbol.unique() if s.isalpha()]

def calculate_volatility(daily_returns):
    daily_volatility = np.std(daily_returns['close'])
    if daily_returns['close'].max()>2000 or daily_returns['close'].min()<5:
      return 0
    else:
      scaled_volatility = (daily_volatility - 0) / (daily_returns['close'].mean())
      return scaled_volatility

def filter_stocks_by_volatility(stocks_data, threshold):
    volatile_stocks = []
    volatility_max = 0
    for stock_name, daily_returns in stocks_data.items():
        volatility = calculate_volatility(daily_returns)
        if volatility > volatility_max:
          volatility_max = volatility
        if volatility >= threshold:
            volatile_stocks.append(stock_name)

    print(volatility_max)
    return volatile_stocks

def lower_circ_stocks(stocks_data, volatile_list,threshold):
  lower_circ = []
  for stock_name in volatile_list:
    if stocks_data[stock_name].iloc[-1].values[0]<stocks_data[stock_name].quantile(threshold/100).values[0]:
      lower_circ.append(stock_name)
  return lower_circ

named_stocks_dict = data[data.tradingsymbol.isin(named_stocks)]
token_mapping = named_stocks_dict[['tradingsymbol','instrument_token']].set_index('instrument_token').to_dict()

stock_data = {}
c = 0
for token in named_stocks_dict.instrument_token.values:
    print(c, token)
    instrument_token = token
    from_datetime = datetime.datetime.now() - datetime.timedelta(days=120)   # From last & days
    to_datetime = datetime.datetime.now()
    interval = "60minute"
    try:
      df = pd.DataFrame(kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False))
      stock_data[list(token_mapping['tradingsymbol'].values())[c]] = df[['close']]
      c+=1
    except Exception as e:
      print(e)
      c+=1
      pass

reverse_token_map = {v: k for k, v in token_mapping['tradingsymbol'].items()}
pd.DataFrame({"token": [reverse_token_map[list(stock_data.keys())[x]] for x in range(len(stock_data.keys()))],"Name":stock_data.keys()}).to_csv('Stock_instrument_keys.csv')

volatility_threshold = .1
volatile_stocks = filter_stocks_by_volatility(stock_data, volatility_threshold)
lower_circuit_stocks = lower_circ_stocks(stock_data,volatile_stocks,20) #divides by 100
print('Nummber of volatile stocks below {} = '.format(volatility_threshold),len(volatile_stocks))
print('Nummber of stocks in lower circuit = ', len(lower_circuit_stocks))

lower_circuit_stocks

"""# Select items from watchlist"""

watchlist = lower_circuit_stocks
watchlist += ['BHEL','TFCILTD','APOLLO']

watch_df = data[data.tradingsymbol.isin(watchlist)]
token_mapping = watch_df[['tradingsymbol','instrument_token']].set_index('instrument_token').to_dict()
token_mapping

"""# Fetch data for items in watchlist - To be run at 10:15 am"""

list_of_df = {}
c = 0
for token in watch_df.instrument_token.values:
    print(token)
    instrument_token = token
    from_datetime = datetime.datetime.now() - datetime.timedelta(days=120)   # From last & days
    to_datetime = datetime.datetime.now()
    interval = "60minute"
    list_of_df[list(token_mapping['tradingsymbol'].values())[c]] = pd.DataFrame(kite.historical_data(instrument_token, from_datetime, to_datetime, interval, continuous=False, oi=False))
    c+=1

"""#@title Regression functions

"""

from math import sqrt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

def pythag(pt1, pt2):
    a_sq = (pt2[0] - pt1[0]) ** 2
    b_sq = (pt2[1] - pt1[1]) ** 2
    return sqrt(a_sq + b_sq)

def regression_ceof(pts):
    X = np.array([pt[0] for pt in pts]).reshape(-1, 1)
    y = np.array([pt[1] for pt in pts])
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_

def local_min_max(pts):
    local_min = []
    local_max = []
    prev_pts = [(0, pts[0]), (1, pts[1])]
    for i in range(1, len(pts) - 1):
        append_to = ''
        if pts[i-1] > pts[i] < pts[i+1]:
            append_to = 'min'
        elif pts[i-1] < pts[i] > pts[i+1]:
            append_to = 'max'
        if append_to:
            if local_min or local_max:
                prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                curr_distance = pythag(prev_pts[1], (i, pts[i]))
                if curr_distance >= prev_distance:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == 'min':
                        local_min.append((i, pts[i]))
                    else:
                        local_max.append((i, pts[i]))
            else:
                prev_pts[0] = prev_pts[1]
                prev_pts[1] = (i, pts[i])
                if append_to == 'min':
                    local_min.append((i, pts[i]))
                else:
                    local_max.append((i, pts[i]))
    return local_min, local_max

support = {}
resistance = {}
midway = {}
for k,v in list_of_df.items(): #here v = dataframe
#     print(k)
    plt.figure(figsize=(5,3))
    v['average'] = round((v['open']+v['close'])/2,1)
    signal = v.close
    signal.index = np.arange(signal.shape[0])
    pts = savgol_filter(signal,4, 3)
    local_min, local_max = local_min_max(pts)
    local_min = local_min[-3:]
    local_max = local_max[-3:]
    plt.plot(signal)
    plt.plot(pts)
    for pt in local_min:
        plt.scatter(pt[0], pt[1], c='r')
    for pt in local_max:
        plt.scatter(pt[0], pt[1], c='g')
    local_min_slope, local_min_int = regression_ceof(local_min)
    local_max_slope, local_max_int = regression_ceof(local_max)
    support[k] = (local_min_slope * np.array(signal.index)) + local_min_int
    resistance[k] = (local_max_slope * np.array(signal.index)) + local_max_int
    support_error = support[k] - 1
    resistance_error = resistance[k] + 1
    midway[k] = (support[k]+resistance[k])/2
    plt.plot(pts)
    plt.plot(support[k], label='Support', c='r')
    plt.plot(support_error,c='r')
    plt.plot(resistance[k], label='Resistance', c='g')
    plt.plot(resistance_error,c='g')
    plt.plot(midway[k],label='Midway',c='b')
    plt.title(k)
    plt.legend()
    plt.show()

"""# Close data from previous hour"""

for k,v in list_of_df.items():
    print(k)
#     print(v.open.tail(1))
    print(v.close.tail(1))

"""# Sell/Buy Data in real-time"""

realtime_price = {}
for token in watch_df.instrument_token.values:
    print(token)
    quote = kite.quote(token)
    if quote[str(token)]['depth']['sell'][0]['price'] !=0:
        realtime_price[token_mapping['tradingsymbol'][token]] = quote[str(token)]['depth']['sell'][0]['price']
    else:
        realtime_price[token_mapping['tradingsymbol'][token]] = quote[str(token)]['depth']['buy'][0]['price']

"""# Checking for placement in plot"""

# negative value means above
for k, v in list_of_df.items():
    margins = [support[k][-1],midway[k][-1],resistance[k][-1]]
    result = [ticker - realtime_price[k] for ticker in margins]
#     print(margins)
#     print(realtime_price[k])
    print(result)

"""## conditions :
- If all positive, point is below the support line
- If all negative, point is above the resistance line
- If >midway but <support, it is in bottom half
- If <midway but > resistance, it is in top half

# LSTM model to predict future price
"""

def adf_test(series):
    result = adfuller(series,autolag='AIC')
    labels = ['ADF test stat','p-val','lags used','observn']
    out = pd.Series(result[0:4],index = labels)
#     print(out,file_obj)
    for key,val in result[4].items():
        print(val)
    if result[1]<=.05:
        print('Strong evidence against null hypothesis\nNull Hypothesis rejected\nData has no unit root and is stationary')
        print('Stationary',file_obj)
    else:
        print('Weak evidence against null hypothesis\nNull Hypothesis failed to reject\nData has a unit root and is not stationary')
        print('Not stationary',file_obj)

def create_dataset(data,look_back=1):
    dataX, dataY = [],[]
    for i in range(len(data)-look_back-1):
        dataX.append(data[i:i+look_back,0])
        dataY.append(data[i+look_back,0])
    return np.array(dataX),np.array(dataY)

import tensorflow.keras.backend as K
def RMSE(y_actual,y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_actual)))

close_pts = []
for k,v in list_of_df.items(): #here v = dataframe
    signal2 = v.close
    signal2.index = np.arange(signal2.shape[0])
    close_pts.append(signal2) #600 points for 5minute
print(len(close_pts[0]))

count = 0
f = open('./text_dump.txt', 'w+')
for i in range(len(close_pts)):
    item = list(token_mapping['tradingsymbol'].values())[count]
    print(item,f)
    date = str(datetime.datetime.now())[0:16]
    #series is sampled by 1 hour
    #so it will predict for the next n hours
    # print('.75 = ', series.quantile(.75))
    # print('.25 = ', series.quantile(.25))
    # plot_acf(series,lags = 7)
    # plt.savefig('ACF_{}'.format(item))
    # plt.close()
    # result = seasonal_decompose(series,model='multiplicative',period = 7)
    # result.plot()
    # plt.savefig('Trend_{}'.format(item))
    # plt.close()
    scaler = MinMaxScaler(feature_range=(0,1))
    close_dataset = scaler.fit_transform(np.array(close_pts[i].values).reshape(-1,1))
    look_back = int(len(close_dataset)/5)
    print('LOOK BACK = ',look_back)
    train = close_dataset
    # test = close_dataset[450:]

    train_X,train_Y = create_dataset(train,look_back)
    # test_X,test_Y = create_dataset(test,look_back)

    train_X = np.reshape(train_X,(train_X.shape[0],train_X.shape[1]))
    train_Y = np.reshape(train_Y,(train_Y.shape[0],1))
    # test_X = np.reshape(test_X,(test_X.shape[0],test_X.shape[1]))
    # test_Y = np.reshape(test_Y,(test_Y.shape[0],1))


    print('training close')
    model = Sequential()
    model.add(LSTM(100,return_sequences = True,input_shape = (look_back,1)))
    model.add(LSTM(look_back))
    model.add(Dense(1))
    model.compile(loss = RMSE, optimizer='adam')
    model.fit(train_X,train_Y,epochs=70,batch_size=24,verbose=1)

    trainPredict = scaler.inverse_transform(model.predict(train_X))
    print(RMSE(scaler.inverse_transform(train_Y),trainPredict),f)

    # testPredict = scaler.inverse_transform(model.predict(test_X))
    # print(RMSE(scaler.inverse_transform(test_Y),testPredict),f)

    print('forecasting')

    x_input_C = close_dataset[len(close_dataset)-look_back:].reshape(1,-1)
    # x_input = testX[len(testX)-look_back:].reshape(1,-1)

    temp_input = list(x_input_C)[0].tolist()
    forecast_C = []
    counter = 0
    intervals_to_predict = look_back
    while(counter<intervals_to_predict):
#         print(forecast)
        if(len(temp_input)>look_back):
            x_input_C=np.array(temp_input[counter:])
            x_input_C=x_input_C.reshape(1,-1)
            x_input_C = x_input_C.reshape((look_back, 1))
            yhat = model.predict(x_input_C, verbose=0)
            temp_input.extend(yhat[0].tolist())
            forecast_C.extend(yhat.tolist())
            counter+=1
        else:
            x_input_C = x_input_C.reshape((look_back,1))
            yhat = model.predict(x_input_C, verbose=0)
            temp_input.extend(yhat[0].tolist())
            forecast_C.extend(yhat.tolist())
            counter+=1

    forecast_C = forecast_C[-intervals_to_predict:]

    # print('Forecast : \n',scaler.inverse_transform(forecast),f)

    df3_C = close_dataset.tolist()
    df3_C.extend(forecast_C)

    output2 = pd.DataFrame({'Date':[x for x in range(len(df3_C))]})
    output2['Historical'],output2['Predicted'],output2['Forecast'] = np.nan,np.nan,np.nan
    output2['Historical'][:len(close_pts[i])] = close_pts[i].values
    output2['Predicted'][look_back+1:len(trainPredict)+look_back+1] = trainPredict.reshape(trainPredict.shape[0])
    output2['Forecast'][len(close_pts[i]):] = [x[0] for x in scaler.inverse_transform(forecast_C[-intervals_to_predict:])]

    output2.set_index('Date',inplace=True)
    # output1.to_csv('forecast_O_{}.csv'.format(item))
    output2.plot(figsize = (10,8),x_compat=True)
    plt.title("{}_{}".format(item, date))
    plt.savefig('Plot_{}_{}.png'.format(item,date))
    plt.close()

    # new = pd.DataFrame(scaler.inverse_transform(close_dataset))
    # new['trainPredicted']=np.nan
    # new['testPredicted']=np.nan
    # new['trainPredicted'][look_back:len(trainPredict)+look_back] = trainPredict.reshape(trainPredict.shape[0])
    # new['testPredicted'][len(trainPredict)+look_back+look_back-1:len(trainPredict)+look_back+len(testPredict)+look_back-1]=testPredict.reshape(testPredict.shape[0])
    # new.plot(title=item)

    count+=1

new = pd.DataFrame(scaler.inverse_transform(close_dataset))
    new['trainPredicted']=np.nan
    new['testPredicted']=np.nan
    new['trainPredicted'][look_back-2:len(trainPredict)+look_back-2] = trainPredict.reshape(trainPredict.shape[0])
    new['testPredicted'][len(trainPredict)+look_back+look_back-1:len(trainPredict)+look_back+len(testPredict)+look_back-1]=testPredict.reshape(testPredict.shape[0])
    new.plot(title=item)
