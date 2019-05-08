import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics
from keras import backend as K

import numpy as np
import decimal
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Model():

    def __init__(self):

        # Number of inputs for training (will loose 1)
        self.eval_lookback  = 100 + 1# input batch size will be eval_lookback + window_len-1
        
        # We will feed in the past n open-to-open price changes
        self.window_len = 10
        
        # How much historical data do we need?
        self.warmup_count = self.eval_lookback + self.window_len

        # Intialise keras model
        self.keras_setup()

    def keras_setup(self):

        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        
        model = Sequential()

        # model.add(LSTM(20, input_shape=(10, 1)))
        # model.add(Dropout(0.25))
        # model.add(Dense(units=1))
        # model.add(Activation("linear"))

        # model.compile(loss="mae", optimizer="adam")

        model.add(Dense(units=256, activation = 'relu', input_dim = self.window_len))
        model.add(Dropout(0.5))
        model.add(Dense(units=128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=64, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=32, activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=2, activation = 'softmax'))
        model.compile(loss='mae', optimizer="adam", metrics=['accuracy'])

        self.y_pred = model

    def preprocessing(self):

        # Create our input feature dataset and corresponding labels
        all_data = np.append(self.hist_data.open.values.flatten().astype(np.float32), self.current_price)
        features   = []
        labels     = []
        for i in range(self.window_len+1, len(all_data)-1):
            
            # input is change in priice
            features.append(np.diff(all_data[i-self.window_len-1:i].copy()))
            
            # label is change in price from last day in input to the next day
            dp = 100.*(all_data[i+1]-all_data[i])/all_data[i]

            # binarise the labels
            if dp > 0.0:
                dp = 1
            else:
                dp = 0
            labels.append(dp)

        self.features = np.array(features)
        self.labels   = np.array(labels)

        # Standardise the features 
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)

        # # Reshape features
        # self.features = self.features.reshape((self.features.shape[0], self.features.shape[1], 1))
        
        # convert to one hot for tensorflow
        oh = np.zeros((len(labels),2))
        oh[np.arange(len(labels)),labels] = 1.0
        self.labels = oh

    def train(self):
        # Fit the model
        with self.session.as_default():
            with self.graph.as_default():
                self.y_pred.fit(self.features, self.labels,epochs=50)
        
    def predict(self):
        
        "make predictions"
        with self.session.as_default():
            with self.graph.as_default():
                pred_feat  =  np.append(self.hist_data.open.values.flatten().astype(np.float32), self.current_price)[-self.window_len-1:]
                pred_feat  = np.diff(pred_feat)
                pred_feat = np.array([pred_feat.tolist()])
                # pred_feat = pred_feat.reshape((pred_feat.shape[0], pred_feat.shape[1], 1))
                pred_proba = self.y_pred.predict(pred_feat, batch_size = 20)
                self.current_forecast = pred_proba[0]
                return np.argmax(pred_proba[0])
       
class Crypto_Trade(QCAlgorithm):
    
    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.Debug("START: Initialize")

        # init the keras model object
        self.model = Model()

        # setup backtest
        self.SetStartDate(2016,1,1)  #Set Start Date
        self.SetEndDate(2018,9,30)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash

        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.currency = "BTCUSD"
        self.SetWarmUp(self.model.warmup_count)

        # Find more symbols here: http://quantconnect.com/data
        self.symbol = self.AddCrypto(self.currency, Resolution.Daily).Symbol
        self.model.symbol = self.symbol

        # Our big history call, only done once to save time
        self.model.hist_data = self.History([self.symbol,], self.model.warmup_count, Resolution.Daily).astype(np.float32)

        # Flag to know when to start gathering history in OnData or Rebalance
        self.do_once = True

        # Prevent order spam by tracking current weight target and comparing against new targets
        self.target = 0.0

        # We are forecasting and trading on open-to-open price changes on a daily time scale. So work every morning.
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), \
            self.TimeRules.AfterMarketOpen(self.symbol), \
            Action(self.Rebalance))


    def OnData(self, data):
        if self.IsWarmingUp:
            return

    def Rebalance(self):

        # store current price for model to use at end of historical data
        self.model.current_price = float(self.Securities[self.symbol].Price)
        
        # Accrew history over time vs making huge, slow history calls each step. # Updates the training set on rolling window basis
        if not self.do_once:
            new_hist      = self.History([self.symbol,], 1, Resolution.Minute).astype(np.float32)
            self.model.hist_data = self.model.hist_data.append(new_hist).iloc[1:] #append and pop stack
        else:
            self.do_once  = False
        
        # Prepare our data now that it has been updated
        self.model.preprocessing()
        
        # Fit the model
        self.model.train()
        
        # Using the latest input feature set, lets get the predicted assets expected to make the desired profit by the next open
        signal = self.model.predict()

        # In case of repeated forecast, lets skip rebalance and reduce fees/orders         
        if signal != self.target:
            
            # track our current target to allow for above filter
            self.target = signal
            # rebalance
            self.SetHoldings(self.symbol, self.target, liquidateExistingHoldings = True)