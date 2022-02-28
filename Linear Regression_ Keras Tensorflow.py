import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, preprocessing
import matplotlib.pyplot as plt 


#Import csv file with data
filename = "C:/Users/eleni antonakaki/Desktop/machine learning ex01/Linear regression/HousingData (1).csv"
data_multi = pd.read_csv(filename,header= None, names =["CRIM","ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD","TAX", "PTRATIO", "B", "LSTAT", "MEDV"]) 

print ("Total number of rows in dataset = {}".format(data_multi.shape[0]))
print ("Total number of columns in dataset = {}".format(data_multi.shape[1]))
print("-----------------------------------------------------------------------")
#new value df
df=data_multi 
#new value in order to fill nan values equal to 0
df_nonnan= df.dropna() 
#new value for removing the first row with headers
headers_del= df_nonnan.drop(df_nonnan.index[0])
headers_del.head()

temp_multi = headers_del.to_numpy(dtype=np.float64)

#features and target values
X_multi = temp_multi[:,:-1] # get vectors for features=X
y_multi = temp_multi[:,-1] # get vectors for output values for MEDV(y)

#print bias
print("Bias: ",temp_multi.intercept_)

#normalize features
X_multi = preprocessing.normalize(X_multi)

X_multi = np.column_stack((np.ones((X_multi.shape[0])),X_multi))

#split data into train and test set
X_train, X_test, y_train, y_test= train_test_split(X_multi,y_multi, test_size=0.10, train_size=0.90, random_state=2)

#create an linear regression object
regr_multi = linear_model.LinearRegression()

#Train my model
regr_multi.fit(X_train, y_train)

#print bias
print("Bias: ",regr_multi.intercept_)
#print coeff's
Coefficients= regr_multi.coef_
print('Coefficients: \n', Coefficients)

print("-----------------------------------------------------------------------")

#make predictions with test set
y_pred = regr_multi.predict(X_test)

#make predictions with train set
y_pred_train = regr_multi.predict(X_train)

#ploting actual target values and predictions
plt.plot(y_test, color='blue')
plt.plot(y_pred, 'o', color='pink')

num_data=X_multi.shape[0]

# The RSE for test set
mse= metrics.mean_squared_error(y_test, y_pred)
print('RSE for test set= ', math.sqrt(mse/(num_data-2)))
# The coefficient of determination for test set
print('R2 for test set= ', metrics.r2_score(y_test,y_pred))

print("-----------------------------------------------------------------------")

# The RSE for train set
mse= metrics.mean_squared_error(y_train, y_pred_train)
print('RSE for train set= ', math.sqrt(mse/(num_data-2)))
# The coefficient of determination for train set
print('R2 for train set= ', metrics.r2_score(y_train,y_pred_train))

print("-----------------------------------------------------------------------")
    

#last task

#1st 

#split data into 50% per train and test set
X_train1, X_test1, y_train1, y_test1= train_test_split(X_multi,y_multi, test_size=0.50, train_size=0.50, random_state=4)
regr_multi = linear_model.LinearRegression()
regr_multi.fit(X_train1, y_train1)
y_pred1 = regr_multi.predict(X_test1)

#make predictions with train set
y_pred_train1 = regr_multi.predict(X_train1)


# The RSE for test set
mse1= metrics.mean_squared_error(y_test1, y_pred1)
print('RSE for test set(1)= ', math.sqrt(mse1/(num_data-2)))
# The coefficient of determination for test set
print('R2 for test set(1)= ', metrics.r2_score(y_test1,y_pred1))

print("-----------------------------------------------------------------------")

# The RSE for train set
mse= metrics.mean_squared_error(y_train1, y_pred_train1)
print('RSE for train set(1)= ', math.sqrt(mse/(num_data-2)))
# The coefficient of determination for train set
print('R2 for train set(1)= ', metrics.r2_score(y_train1,y_pred_train1))

print("-----------------------------------------------------------------------")


#2nd

#split data into 34% for test set and 66% for train set
X_train2, X_test2, y_train2, y_test2= train_test_split(X_multi,y_multi, test_size=0.34, train_size=0.66, random_state=6)
regr_multi = linear_model.LinearRegression()
regr_multi.fit(X_train2, y_train2)
y_pred2 = regr_multi.predict(X_test2)

#make predictions with train set
y_pred_train2 = regr_multi.predict(X_train2)

# The RSE for test set
mse2= metrics.mean_squared_error(y_test2, y_pred2)
print('RSE for test set(2)= ', math.sqrt(mse2/(num_data-2)))
# The coefficient of determination for test set
print('R2 for test set(2)= ', metrics.r2_score(y_test2,y_pred2))

print("-----------------------------------------------------------------------")

# The RSE for train set
mse2= metrics.mean_squared_error(y_train2, y_pred_train2)
print('RSE for train set(2)= ', math.sqrt(mse2/(num_data-2)))
# The coefficient of determination for train set
print('R2 for train set(2)= ', metrics.r2_score(y_train2,y_pred_train2))

print("-----------------------------------------------------------------------")

#3rd

#split data into 5% for test set and 95% for train set
X_train3, X_test3, y_train3, y_test3= train_test_split(X_multi,y_multi, test_size=0.05, train_size=0.95,random_state=9)
regr_multi = linear_model.LinearRegression()
regr_multi.fit(X_train3, y_train3)
y_pred3 = regr_multi.predict(X_test3)

#make predictions with train set
y_pred_train3 = regr_multi.predict(X_train3)


# The RSE for test set
mse3= metrics.mean_squared_error(y_test3, y_pred3)
print('RSE for test set(3)= ', math.sqrt(mse3/(num_data-2)))
# The coefficient of determination for test set
print('R2 for test set(3)= ', metrics.r2_score(y_test3,y_pred3))

print("-----------------------------------------------------------------------")

# The RSE for train set
mse3= metrics.mean_squared_error(y_train3, y_pred_train3)
print('RSE for train set(3)= ', math.sqrt(mse3/(num_data-2)))
# The coefficient of determination for train set
print('R2 for train set(3)= ', metrics.r2_score(y_train3,y_pred_train3))













