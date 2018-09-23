
# coding: utf-8

# In[4]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
py.init_notebook_mode(connected=True)
get_ipython().magic('matplotlib inline')


# In[5]:


import matplotlib
import pylab
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import cm as cm
import calendar
import warnings
import itertools
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sb
import itertools
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from numpy import loadtxt
import os 
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")


y = 2016
new_data = pd.DataFrame()
sample_times = []

for y in range(2014,2016,1):
   print (y)
   for m in range(1,13,1):
       no_of_days = calendar.monthrange(2014,m)[1]

       for d in range (1,no_of_days+1,1):
##        for d in range (1,2,1):

      #      data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
           data = pd.read_csv("F:\edit\Data\data\Predicting\\D120318_%d%02d%02d_0000.csv"%(y,m,d));
           pd
           
           if (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==600:
               new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
               new_data = new_data.append(new_data_temp)

               for i in range(len(new_data_temp)):
                   sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=10))
       
           elif (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==60:
               new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:1440].copy()
               new_data = new_data.append(new_data_temp)

               for i in range(len(new_data_temp)):
                   sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=1))

new_data.columns=['time','wind_speed','wind_dir','humidity','temperature','pressure','dhi','ghi','voltage']
sample_times_series = pd.Series(sample_times)
new_data['time'] = sample_times_series.values
new_data = new_data.reset_index().set_index('time').resample('10min').mean()


# In[8]:


new_data.drop('index', axis=1, inplace=True)


# In[9]:


new_data.drop('ghi', axis=1, inplace=True)


# In[10]:


new_data.drop('voltage', axis=1, inplace=True)


# In[14]:


btc_trace = go.Scatter(x=new_data.index, y=new_data['dhi'], name= 'Price')
py.iplot([btc_trace])


# In[15]:


new_data['dhi'].replace(0, np.nan, inplace=True)
new_data['dhi'].fillna(method='ffill', inplace=True)


# In[16]:


btc_trace = go.Scatter(x=new_data.index, y=new_data['dhi'], name= 'dhi')
py.iplot([btc_trace])


# In[23]:


from sklearn.preprocessing import MinMaxScaler
values = new_data['dhi'].values
np.random.seed(1)
values = values.reshape(-1,1)


# In[26]:


scaler = MinMaxScaler(feature_range=(0, 1))


# In[28]:


y=new_data

y = y.fillna(y.bfill())
values =y.values


# In[29]:


scaled = scaler.fit_transform(values)


# In[30]:


train_size = int(len(scaled) * 0.75)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))


# In[31]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# In[32]:


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[33]:


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[34]:


model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)


# In[35]:


pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[36]:


yhat = model.predict(testX)
pyplot.plot(yhat, label='predict')
pyplot.plot(testY, label='true')
pyplot.legend()
pyplot.show()


# In[62]:


yhat = model.predict(testX)


# In[66]:


# Let`s import all packages that we may need:

import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


# In[67]:


new_data.columns


# In[71]:


## finding all columns that have nan:

droping_list_all=[]
for j in range(0,6):
    if not new_data.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
droping_list_all


# In[73]:


for j in range(0,6):        
        new_data.iloc[:,j]=new_data.iloc[:,j].fillna(new_data.iloc[:,j].mean())


# In[74]:


new_data.isnull().sum()


# In[75]:


new_data.dhi.resample('D').sum().plot(title='DHI resampled over day for sum') 
#df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day', color='red') 
plt.tight_layout()
plt.show()   
new_data.dhi.resample('D').mean().plot(title='DHI resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[77]:


# Below I compare the mean of different featuresresampled over day. 
# specify columns to plot
cols = [0, 1, 2, 3, 4, 5]
i = 1
groups=cols
values = new_data.resample('D').mean().values
# plot each column
plt.figure(figsize=(15, 10))
for group in groups:
	plt.subplot(len(cols), 1, i)
	plt.plot(values[:, group])
	plt.title(new_data.columns[group], y=0.75, loc='right')
	i += 1
plt.show()


# In[78]:


new_data.dhi.resample('W').mean().plot(color='y', legend=True)
plt.show()


# In[79]:


new_data.wind_speed.resample('W').mean().plot(color='r', legend=True)


# In[80]:


new_data.humidity.resample('W').mean().plot(color='b', legend=True)


# In[81]:


new_data.temperature.resample('W').mean().plot(color='g', legend=True)


# In[82]:


new_data.pressure.resample('W').mean().plot(color='y', legend=True)


# In[84]:


new_data.wind_dir.resample('W').mean().plot(color='r', legend=True)


# In[85]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[86]:


## resampling of data over hour
df_resample = new_data.resample('h').mean() 
df_resample.shape


# In[88]:


## * Note: I scale all features in range of [0,1].

## If you would like to train based on the resampled data (over hour), then used below
values = df_resample.values 


## full data without resampling
#values = df.values

# integer encode direction
# ensure all data is float
#values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict

print(reframed.head())


# In[89]:


# split into train and test sets
values = reframed.values

n_train_time = 365*24
train = values[:n_train_time, :]
test = values[n_train_time:, :]
##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].


# In[92]:


model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')



# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=70, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



# In[122]:


yhat = model.predict(test_X)


# In[128]:


test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# In[132]:


inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)


# In[130]:


test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)


# In[131]:


rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[140]:


inv_yhat = np.concatenate((yhat, test_X[:, -5:]), axis=1)


# In[141]:


inv_yhat = scaler.inverse_transform(inv_yhat)


# In[142]:


inv_yhat = inv_yhat[:,0]


# In[143]:


test_y = test_y.reshape((len(test_y), 1))


# In[144]:


inv_y = np.concatenate((test_y, test_X[:, -5:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# In[145]:


rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

