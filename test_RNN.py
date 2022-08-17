
#importing necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

## function to pre-process test data
def pre_test_data(data):

	# loading x and y data
	test_X = data.drop('Target', axis = 1)
	test_y = data.loc[:,['Target']]

	sc_input = pickle.load(open("./data/input_train_val.pkl", 'rb'))
	sc_output = pickle.load(open("./data/output_train_val.pkl", 'rb'))

	# using loaded scaled values to test data
	sc_x_test = sc_input.transform(test_X)
	sc_y_test = sc_output.transform(test_y)

	# reshaping the x values to 4 features to each
	final_X_train = np.reshape(sc_x_test,(sc_x_test.shape[0],3,4))
	final_y_train = sc_y_test

	# transforming y values
	final_y_test = sc_output.inverse_transform(final_y_train)
	final_X_test = final_X_train	

	return final_X_test, final_y_test, sc_output

##function to predict values using model
def predict(mod, test_data, scaled_out):
    pred_val = mod.predict(test_data)
    pred_val = scaled_out.inverse_transform(pred_val)
    return pred_val

##defining fucntion to calculate error
def error_pred(predict_val, real_val):
	err = predict_val - real_val
	mse = mean_squared_error(predict_val, real_val)
	#rmse = np.sqrt(mse)
	mae = np.abs(err).mean()
	return mse, mae


## function to plot true vs predicted values
def plot(pre_model, y_test_val):
	plt.figure(figsize=(15,8))
	plt.title("Plot for actual vs predicted opening price")
	plt.plot(y_test_val,label='Actual Price',color='b',marker="+")
	plt.plot(pre_model,label='Predicted Price',color='m',marker="X")
	plt.xlabel('Time (day)')
	plt.ylabel('Price ($)')
	plt.legend()
	plt.show()


##calling main function
if __name__ == "__main__":

	#loading the saved trained model 
	lstm_model = keras.models.load_model("./models/Group_24_RNN_model.h5") # Load your saved model
	 
	#reading saved test data  
	t_data = pd.read_csv('./data/test_data_RNN.csv') # Load testing data
	
	#calling function to pre-process 
	pre_xtest, pre_ytest, scaled_output = pre_test_data(t_data)

	#predicted values of the model
	predict_lstm = predict(lstm_model, pre_xtest, scaled_output)

	#calculating error
	mse, mae = error_pred(predict_lstm, pre_ytest) 
	print('Mean Absolute Error: {:.4f}'.format(mae))
	
	print('Total Test data loss i.e. Mean squared error is: {:.4f}'.format(mse))
	
	#plotting predicted values
	plot(predict_lstm, pre_ytest) 


