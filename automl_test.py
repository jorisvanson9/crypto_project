import sys
import os
import time

import pandas as pd
import numpy as np
import talib
import h2o
from h2o.automl import H2OAutoML
from pysparkling import *

# Now we are ready to import Spark Modules:
from pyspark.sql import SparkSession
from pyspark import SparkConf

from automl_crypto_backtest import automl_crypto_backtest

print("Successfully imported all Spark stuff")

os.chdir('/home/jorisvanson/PycharmProjects/crypto_project_new')

os.environ['PYSPARK_PYTHON'] = '/home/jorisvanson/anaconda3/envs/crypto_project_new/bin/python3.6'

# spark_config = SparkConf().setMaster('spark://jorisvanson-Latitude-E5570:7077')\
#     .setAppName('H2O APP').set('spark.executor.memory', '4g').set('spark.executor.cores', '3')\
#     .set('spark.scheduler.minRegisteredResourcesRatio', '1')

spark_config = SparkConf().setMaster('local[*]')\
    .setAppName('H2O APP').set('spark.executor.memory', '4g').set('spark.executor.cores', '3')\
    .set('spark.scheduler.minRegisteredResourcesRatio', '1')

# spark_config = SparkConf().setMaster('spark://192.168.176.25:7077')\
#     .setAppName('H2O CRYPTO APP').set('spark.executor.memory', '4g').set('spark.executor.cores', '3')\
#     .set('spark.scheduler.minRegisteredResourcesRatio', '1')

sc = SparkSession.builder.config(conf=spark_config).getOrCreate()

hc = H2OContext.getOrCreate(sc)

symbol = 'LTC-EUR'

days = 24

data = pd.read_feather('./ltc_eur_data.feather')

data['returns_day'] = data['close']/data['close'].shift(days)-1

data['ind'] = np.where(data['returns_day'] > 0.005, 1, 0)


macd, macdsignal, macdhist = talib.MACDFIX(data['close'], signalperiod=90)

data['MACD'] = macd.shift(days) - macdsignal.shift(days)
data['MACD_signal'] = macdsignal.shift(days)

data['AD'] = talib.AD(data['high'], data['low'], data['close'], data['volume']).shift(days)

data['OBV'] = talib.OBV(data['close'], data['volume']).shift(days)

data['RSI'] = talib.RSI(data['close'], timeperiod=200).shift(days)

data['NATR'] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=200).shift(days)

upperband, middleband, lowerband = talib.BBANDS(data['close'], timeperiod=200, nbdevup=2, nbdevdn=2, matype=0)

data['BB_UPPER'] = upperband.shift(days)
data['BB_MIDDLE'] = middleband.shift(days)
data['BB_LOWER'] = lowerband.shift(days)

output = data[['time', 'ind', 'MACD', 'RSI', 'NATR', 'OBV', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']].dropna()

output = output[output['time'] < '2018-09-01']

output = h2o.H2OFrame(output)
train, test = output.split_frame(ratios=[.8])

# Identify predictors and response
x = train.columns
y = "ind"
x.remove(y)
x.remove("time")

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 60 seconds
# aml = H2OAutoML(max_runtime_secs=120, exclude_algos=['StackedEnsemble'])
# max_models = 50
aml = H2OAutoML(max_runtime_secs=120)

aml.train(x=x, y=y, training_frame=train, leaderboard_frame=test)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)
# aml.leader

# Save leader model as H2O object
h2o.save_model(aml.leader, './' + str(symbol) + '_bin_model')

# Save leaderboard as text file
lb_output = lb.as_data_frame()
lb_output.to_csv('./' + str(symbol) + '_bin_model/' + str(symbol) + '_' + str(days) + '_leaderboard.txt')

automl_crypto_backtest(aml.leader)