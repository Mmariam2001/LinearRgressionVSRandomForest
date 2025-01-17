# LinearRgressionVSRandomForest
LinearRgressionVSRandomForest

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dvd = pd.read_csv("rental_info.csv")
#print(dvd.info())
print(dvd[['rental_date','return_date']])


#Since the type for the "return_date" and "rental_date" is object 
# We need to change it to to_date_time()

dvd= pd.DataFrame(dvd)
dvd['rental_date']= pd.to_datetime(dvd['rental_date'])
dvd['return_date']= pd.to_datetime(dvd['return_date'])

#add the new column "rental_length_days"
dvd['rental_length_days'] = (dvd['return_date'] - dvd['rental_date']).dt.days

print(dvd['rental_length_days'])

# get_dummies special features part
behind_the_scenes= []
deleted_scenes= []
#print(dvd['special_features'])
for features in dvd['special_features']:
    if "Behind the Scenes" in features:
        behind_the_scenes.append(1)
    else:
        behind_the_scenes.append(0)
        
    if "Deleted Scenes" in features:
        deleted_scenes.append(1)
    else:
        deleted_scenes.append(0)
dvd['deleted_scenes']= deleted_scenes
dvd['behind_the_scenes']= behind_the_scenes


# choosing the features columns & the label column as X and y
cols_to_drop = ["special_features", "return_date", "rental_length_days", "rental_date"]
X = dvd.drop(cols_to_drop, axis=1)
y = dvd['rental_length_days']

print(X.dtypes)
#splitting the data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


# Create the Lasso model
lasso = Lasso(alpha=0.3, random_state=9) 

# Train the model and access the coefficients
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
X_train_lasso = X_train.iloc[:, lasso_coef > 0]
X_test_lasso = X_test.iloc[:, lasso_coef > 0 ]


# Build Linear Regression Model
LinearR = LinearRegression()
LinearR.fit(X_train_lasso, y_train)
y_pred = LinearR.predict(X_test_lasso)
mse_LinearRegression = mse(y_test, y_pred)
print(mse_LinearRegression)


#preparing numbers of trees & depth
parameters= {"n_estimators": np.arange(1,51,1),
            "max_depth": np.arange(1,6,1)}

#building RandomForestRegressor
rf= RandomForestRegressor() # to use it in CV

#Building crossvalidation --RandomizedSearchCV--
CV= RandomizedSearchCV(rf, param_distributions= parameters, cv=4, random_state=1)
CV.fit(X_train, y_train)

#To find the best parameter using best_param
#this will include best no. of trees & best maximum depth
CV_best_params= CV.best_params_  
print(CV_best_params)

#building the RandomForestRegressor with the best parameters 
#that have been found using RandomizedSearchCV
rm= RandomForestRegressor(n_estimators= CV_best_params['n_estimators'],
                          max_depth= CV_best_params['max_depth'],
                          random_state=1)
rm.fit(X_train, y_train)
rm_y_pred = rm.predict(X_test)
RF_mse= mse(y_test, rm_y_pred)
print(RF_mse)
