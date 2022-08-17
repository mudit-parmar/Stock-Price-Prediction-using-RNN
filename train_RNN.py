
#import necessary libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.layers import LSTM

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GRU, Dropout
import pickle
import warnings
warnings.filterwarnings('ignore')

# to execute the dataset creation code please uncomment this block of code
'''
##Code to create new dataset
#reading the given csv file
previous_data = pd.read_csv('./data/q2_dataset.csv', parse_dates=['Date']) 
# sort data according to the date
previous_data.iloc[:] = previous_data.iloc[::-1].values 

## creating new dataset
def new_dataset(data, previous_values):
#creating new values with target values
    new_X_values, new_Y_values = [], []
    for i in range(len(data)-previous_values):
        v = []
        for p in range(i, i+previous_values):
            for q in range(2,6):
                v.append(data.iloc[p, q])
        new_X_values.append(v)
        new_Y_values.append(data.iloc[i + previous_values, 3])
    return np.array(new_X_values), np.array(new_Y_values)

#using last 3 days as features
latest_days = 3
new_data, new_label = new_dataset(previous_data, latest_days)

# splitting the 70% data into training and rest into testing, randomly.
X_train, X_test, y_train, y_test = train_test_split(new_data, new_label, test_size=0.30, random_state=42)

# saving the splitting data into training and testing csv using appropriate column names
RNN_train = pd.DataFrame(np.c_[X_train, y_train], columns=['Volume-3','Open-3','High-3','Low-3','Volume-2','Open-2','High-2','Low-2','Volume-1','Open-1','High-1','Low-1','Target'])
RNN_test = pd.DataFrame(np.c_[X_test, y_test], columns=['Volume-3','Open-3','High-3','Low-3','Volume-2','Open-2','High-2','Low-2','Volume-1','Open-1','High-1','Low-1','Target'])
RNN_train.to_csv('./data/train_data_RNN.csv', index=False)
RNN_test.to_csv('./data/test_data_RNN.csv', index=False)
'''

##function to pre-process the training data
def pre_train_data(data):
	# splitting the data into x and y
	train_X = data.drop('Target', axis = 1)
	train_y = data.loc[:,['Target']]

	# Calling MinMax function to scale the data
	Min_Max_x = MinMaxScaler(feature_range = (0,1))
	Min_Max_y = MinMaxScaler(feature_range = (0,1))

	# fitting the x and y values to the scaled variable
	scaled_input = Min_Max_x.fit(train_X)
	scaled_output = Min_Max_y.fit(train_y)

	# transforming the training data
	n_xtrain = scaled_input.transform(train_X)
	n_ytrain = scaled_output.transform(train_y)

	# dumping the scaled data to use it further in test data
	pickle.dump(scaled_input, open("./data/input_train_val.pkl", 'wb'))
	pickle.dump(scaled_output, open("./data/output_train_val.pkl", 'wb'))

	# reshaping data into 3-D with 4 features for each day
	final_X_train = n_xtrain.reshape(n_xtrain.shape[0],3,4)
	final_y_train = n_ytrain

	return final_X_train, final_y_train

'''
##creating lstm model to train the data
def model_lstm(train_data):
    model = Sequential()
    #adding 1st layer in lstm
    model.add(LSTM(units = 200, return_sequences = True, input_shape = (train_data.shape[1], train_data.shape[2])))
    #using dropout to randomly update hidden units to 0.
    model.add(Dropout(rate = 0.2))

    #adding 2nd layer to lstm model
    model.add(LSTM(units = 200, return_sequences = False))
    model.add(Dropout(rate = 0.2))
    
    #adding output layer to the model
    model.add(Dense(units = 1))

    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
    model.summary()
    return model
'''


##creating another lstm model
def model_lstm2(train_data):
    model = Sequential()
    
    model.add(LSTM(units = 300, return_sequences = True, input_shape = (train_data.shape[1], train_data.shape[2])))
    model.add(Dropout(rate = 0.2))

    model.add(LSTM(units = 380, return_sequences = False))
    
    model.add(Dense(units = 1))

    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
    model.summary()
    return model


'''
##creating GRU model to test with data
def model_gru(train_data_gru):
    model = Sequential()
    model.add(GRU(units = 64,return_sequences = True, input_shape = [train_data_gru.shape[1], train_data_gru.shape[2]]))
    model.add(GRU(units = 64,return_sequences=True))
    model.add(GRU(units = 64,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae']) # Compile model
    return model
'''

##function to fit the training values to the used model
def assign_to_model(model, X_tr, y_tr):
    #fitting training data to lstm model
    lstm_history = model.fit(X_tr, y_tr, epochs=1000, batch_size=64)
    #fitting in GRU model
    #history_gru = model.fit(X_tr, y_tr, epochs=200, validation_split=0.2, batch_size=32)
    return lstm_history
    #return history_gru

##calling main function
if __name__ == "__main__":
    #reading the given csv data
    read_data = pd.read_csv('./data/train_data_RNN.csv') 
    #pre-processed x and y values 
    pre_xtrain, pre_ytrain = pre_train_data(read_data) 

    #calling appropriate model to use

    #model_value = model_lstm(pre_xtrain) 
    model_value = model_lstm2(pre_xtrain)
    #model_value = model_gru(pre_xtrain)
    
    print('Model Running')

    #using pre-processed data in the model
    history_lstm = assign_to_model(model_value, pre_xtrain, pre_ytrain) 
    #showing training loss
    print('Training loss is : {:.4f}'.format(history_lstm.history['loss'][-1]))
    #saving model 
    model_value.save("./models/Group_24_RNN_model.h5") # Save model


##plot for visulaizing loss in training data
def plot_loss(history):
  #training loss
  loss_train=history.history['loss']
  plt.plot(loss_train,label='Training loss')
  plt.title('LSTM-2')
  plt.ylabel('loss')
  plt.xlabel('no of epoch')
  plt.legend()
  plt.show()


plot_loss(history_lstm)

