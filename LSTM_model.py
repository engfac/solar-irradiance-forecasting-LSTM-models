import math
from matplotlib import pyplot
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import calendar
import itertools
import os 
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
sns.set_style('whitegrid')
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
           data = pd.read_csv("C:\\Users\\USER\\OneDrive - University of Jaffna\\AhilanDoc\\Research\\github\\solar-irradiance-forecasting-ARIMA-models\\data\\D120318_%d%02d%02d_0000.csv"%(y,m,d));
           
           if (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==600:
               new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
               new_data = pd.concat([new_data,new_data_temp])

               for i in range(len(new_data_temp)):
                   sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=10))
       
           elif (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==60:
               new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:1440].copy()
               new_data = pd.concat([new_data,new_data_temp])

               for i in range(len(new_data_temp)):
                   sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=1))

new_data.columns=['time','wind_speed','wind_dir','humidity','temperature','pressure','dhi','ghi','voltage']
sample_times_series = pd.Series(sample_times)
new_data['time'] = sample_times_series.values

new_data = new_data.reset_index().set_index('time').resample('10min').mean()
new_data.drop('index', axis=1, inplace=True)
new_data.drop('ghi', axis=1, inplace=True)
new_data.drop('voltage', axis=1, inplace=True)

#fig = go.Figure(data=go.Scatter(x=new_data.index, y=new_data['dhi'], name= 'dhi'))
#fig.show()

new_data['dhi'].replace(0, np.nan, inplace=True)
new_data['dhi'].fillna(method='ffill', inplace=True)

#fig = go.Figure(data=go.Scatter(x=new_data.index, y=new_data['dhi'], name= 'dhi'))
#fig.show()


values = new_data['dhi'].values
np.random.seed(1)
values = values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

y=new_data
y = y.fillna(y.bfill())
scaled = scaler.fit_transform(y.values)

train_size = int(len(scaled) * 0.75)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



droping_list_all=[]
for j in range(0,6):
    if not new_data.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
droping_list_all



for j in range(0,6):        
        new_data.iloc[:,j]=new_data.iloc[:,j].fillna(new_data.iloc[:,j].mean())


new_data.isnull().sum()



#new_data.dhi.resample('D').sum().plot(title='DHI resampled over day for sum') 
#pyplot.tight_layout()
#pyplot.show()   
#new_data.dhi.resample('D').mean().plot(title='DHI resampled over day for mean', color='red') 
#pyplot.tight_layout()
#pyplot.show()



# Below I compare the mean of different featuresresampled over day. 
# specify columns to plot
#cols = [0, 1, 2, 3, 4, 5]
#i = 1
#groups=cols
#values = new_data.resample('D').mean().values
## plot each column
#pyplot.figure(figsize=(15, 10))
#for group in groups:
#	pyplot.subplot(len(cols), 1, i)
#	pyplot.plot(values[:, group])
#	pyplot.title(new_data.columns[group], y=0.75, loc='right')
#	i += 1
#pyplot.show()


#new_data.dhi.resample('W').mean().plot(color='y', legend=True)
#new_data.wind_speed.resample('W').mean().plot(color='r', legend=True)
#new_data.humidity.resample('W').mean().plot(color='b', legend=True)
#new_data.temperature.resample('W').mean().plot(color='g', legend=True)
#new_data.pressure.resample('W').mean().plot(color='y', legend=True)
#new_data.wind_dir.resample('W').mean().plot(color='r', legend=True)

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


## resampling of data over hour
df_resample = new_data.resample('h').mean() 
df_resample.shape
values = df_resample.values 


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_time = 365*24
train = values[:n_train_time, :]
test = values[n_train_time:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].


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
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper right')
pyplot.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)


test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
