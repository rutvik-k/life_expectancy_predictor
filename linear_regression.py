#things to do
#draw linear regression graph
#predict for new dummy data
#compute r2 and rmse
#find scores fo cross validation
#predict using both laso and reidge regression
#draw gaph for ridge regression for different values of alpha




import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


df = pd.read_csv('gapminder.csv')
y = df['life'].values
X =df.drop(['life','Region'], axis = 1).values

#drawing linear regression line using one feature i.e the fertility rate
y = y.reshape(-1,1)
#print(y)
X_fertility = df['fertility'].values
X_fertility = X_fertility.reshape(-1,1)
plt.scatter(X_fertility,y,color = 'red')

reg = LinearRegression()
reg.fit(X_fertility,y)
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
y_pred = reg.predict(prediction_space)
#print(y_pred)
score = reg.score(X_fertility, y)
print("the simple score of data for single feature without splitting is : " + str(score))

plt.plot(prediction_space,y_pred, color = 'blue' , linewidth = 3)
plt.savefig('liner regression line.png')




#prediction for a dummy data using single feature. Data in testdata.csv file

df_dummy = pd.read_csv('testdata.csv',header=None)
X_dummy = df_dummy.iloc[:,1:2].values
y_dummy = df_dummy.iloc[:,7:8].values
predict_dummy=reg.predict(X_dummy)
print("predicted value when only 1 feature is used is :" + str(predict_dummy))

#now we will split the data and calculate the score(r^2) and rme for the data

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.3, random_state = 42)
reg_all = LinearRegression()
reg_all.fit(X_train,y_train)
X_dummy_all = df_dummy.iloc[:,[0,1,2,3,4,5,6,8]]
y_pred_all = reg_all.predict(X_dummy_all)
print('predicted value using all the features and using linear regression is :' +str(y_pred_all))
y_pred = reg_all.predict(X_test)
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#the following code calculates the cross validation scores for cv = 5
cv_scores = cross_val_score(reg,X,y, cv=5)
# Print the 5-fold cross-validation scores
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))



#regularised linear regression

#we will use 2 types of linear regression, ridge and lasso

#lasso regression. Here we will try to predict the data from testdata.csv with lasso regression. Alo we will draw a graph to show the selection of features with lasso regression

lasso = Lasso(alpha = 0.4, normalize = True)
lasso.fit(X,y)
y_pred_all = lasso.predict(X_dummy_all)
print('predicted value using all the features and using lasso regression is : '+str(y_pred_all))

#code to draw the graph
plt.clf()
lasso_coef = lasso.coef_

df_columns = np.array(['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP','BMI_female', 'child_mortality'])

plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns, rotation=60)
plt.margins(0.02)
plt.savefig('lassofig.png')


#ridge regression
#we will predict the values with ridge regression. We will also plot the graph of score of the model and it's standard deviation using cross validation =10


ridge = Ridge(alpha=0.4, normalize = True)
ridge.fit(X,y)
y_pred_all = ridge.predict(X_dummy_all)
print('predicted value using all the features and using ridge regression is : '+str(y_pred_all))

#now we will draw the graph for different value of alpha with cv=10

alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

ridge = Ridge(normalize = True)


for alpha in alpha_space:

    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge,X,y, cv = 10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.savefig('ridgefig.png')

display_plot(ridge_scores, ridge_scores_std)