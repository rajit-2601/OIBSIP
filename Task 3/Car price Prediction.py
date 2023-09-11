# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fhkH45iUJWgRNgm7uSSMXQ9w3J6g66HT
"""

# Commented out IPython magic to ensure Python compatibility.
#import library
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install keras-tuner
import keras_tuner
import sklearn
# %matplotlib inline
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from math import sqrt
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('CarPrice_Assignment.csv')
df.head()

df.shape

df.info()

df.duplicated('car_ID').sum()

df.describe()

df = df.drop(['car_ID'], axis=1)

df.isnull().sum()

sns.pairplot(y_vars = ['symboling'], x_vars = ['price'] ,data = df)

plt.hist(df["symboling"], bins=10, rwidth=0.8)

df['CarName'].value_counts()
df['car_company'] = df['CarName'].apply(lambda x:x.split(' ')[0])
df['car_company'].head()

df = df.drop(['CarName'], axis =1)

df['car_company'].value_counts()

df['car_company'].replace('toyouta', 'toyota',inplace=True)
df['car_company'].replace('Nissan', 'nissan',inplace=True)
df['car_company'].replace('maxda', 'mazda',inplace=True)
df['car_company'].replace('vokswagen', 'volkswagen',inplace=True)
df['car_company'].replace('vw', 'volkswagen',inplace=True)
df['car_company'].replace('porcshce', 'porsche',inplace=True)

df['car_company'].value_counts()

def num(x):
    return x.map({'four':4, 'two': 2})

df['doornumber'] = df[['doornumber']].apply(num)
df[['doornumber']].head()

sns.stripplot(df['wheelbase'])
plt.show()

plt.hist(df["wheelbase"], bins=25, rwidth=0.8)
plt.show()

df['carlength'].value_counts().head()

sns.stripplot(df['carlength'])
plt.show()

plt.hist(df["carlength"], bins=10, rwidth=0.8)
plt.show()

df['cylindernumber'].value_counts()
df['cylindernumber'].head()

def conv_num(x):
    return x.map({'four': 4,
                  'six': 6,
                  'five': 5,
                  'eight': 8,
                  'two': 2,
                  'three': 3,
                  'twelve': 12}
                 )

df['cylindernumber'] = df[['cylindernumber']].apply(conv_num)
df['cylindernumber'].head()

cars_numeric = df.select_dtypes(include =['int64','float64'])
cars_numeric.head()

plt.figure(figsize = (3,10))
sns.pairplot(cars_numeric)
plt.show()

plt.figure(figsize = (20,20))
sns.heatmap(df[['symboling','doornumber','wheelbase','carlength','carwidth','carheight','curbweight','cylindernumber','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']].corr(), annot = True ,cmap = 'YlGnBu')
plt.show()

cat_cols = df.select_dtypes(include=['object'])
cat_cols.head()

plt.figure(figsize=(20, 12))

plt.subplot(3, 3, 1)
sns.boxplot(x='fueltype', y='price', data=df)

plt.subplot(3, 3, 2)
sns.boxplot(x='aspiration', y='price', data=df)

plt.subplot(3, 3, 3)
sns.boxplot(x='carbody', y='price', data=df)

plt.subplot(3, 3, 4)
sns.boxplot(x='drivewheel', y='price', data=df)

plt.subplot(3, 3, 5)
sns.boxplot(x='enginelocation', y='price', data=df)

plt.subplot(3, 3, 6)
sns.boxplot(x='enginetype', y='price', data=df)

plt.subplot(3, 3, 7)
sns.boxplot(x='fuelsystem', y='price', data=df)

plt.figure(figsize=(20, 10))
sns.boxplot(x='car_company', y='price', data=df)

cars_dum = pd.get_dummies(df[cat_cols.columns])

car_df = pd.concat([df, cars_dum], axis=1)

car_df = car_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
                      'enginetype', 'fuelsystem', 'car_company'], axis=1)

car_df=car_df.drop("cylindernumber",axis=1)
car_df.fillna(method="ffill")

df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)

df_train.shape

df_test.shape

cars_numeric.columns

col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight',  'enginesize', 'boreratio',
            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']

scaler = StandardScaler()
df_train[col_list] = scaler.fit_transform(df_train[col_list])
df_train.describe()

y_train = df_train.pop('price')
X_train = df_train

model = LinearRegression()
model = model.fit(X_train, y_train)

model.intercept_

model.coef_

df_test[col_list] = scaler.transform(df_test[col_list])
y_test = df_test.pop('price')
X_test = df_test
y_pred = model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test,y_pred)
mse

r_squ = r2_score(y_test,y_pred)
r_squ

LRmodel = LinearRegression()
rfe = RFE(LRmodel, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, y_train)

rfe.n_features_

list(zip(X_train.columns, rfe.support_ ,rfe.ranking_))

cols = X_train.columns[rfe.support_]
X_test2 = X_test[cols]
y_pred2 = rfe.predict(X_test)

mse = sklearn.metrics.mean_squared_error(y_test, y_pred2)
print("The Mean Squares Error is: ", mse)
print("The R-2 score is: ", r2_score(y_test, y_pred2))
print("The previous Mean Squares Error was: 4.933082637798508e+23")
print("The previous R-2 score was: -4.263483995248783e+23")

LRmodel = LinearRegression()
rfe = RFE(LRmodel, n_features_to_select=10, step=1)
rfe = rfe.fit(X_train, y_train)
rfe.n_features_
y_pred_ = rfe.predict(X_test)


mse = sklearn.metrics.mean_squared_error(y_test, y_pred_)
print("The Mean Squares Error is: ", mse)
print("The R-2 score is: ", r2_score(y_test, y_pred_))

cols = X_train.columns[rfe.support_]
cols

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

[value for index,value in enumerate(list(zip(X_train.columns,rfe.support_,rfe.ranking_))) if value[2] == True]

df.select_dtypes(include="object")

plt.figure(figsize=(15,6))
sns.histplot(df['price'],color="red",kde=True)
plt.title("Car Price Histogram",fontweight="black",pad=20,fontsize=20)

plt.figure(figsize=(6,4))
counts = df["fueltype"].value_counts()
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Fuel Type",fontsize=15)
plt.ylabel("Number of cars",fontsize=15)
plt.title("Total Cars produced by Fuel Type", pad=20, fontweight="black", fontsize=20)
plt.xticks(rotation=90)
plt.show()

df.nunique()

df = df.drop(["symboling","compressionratio","stroke"],axis=1)
df["fuelsystem"] =  df["fuelsystem"].replace(['mpfi', 'mfi'], "fi")
df["fuelsystem"] =  df["fuelsystem"].replace(['1bbl', '2bbl','4bbl'], "bbl")
df["fuelsystem"] =  df["fuelsystem"].replace(['spdi', 'idi','4bbl'], "spfi")

z = round(df.groupby(["car_company"])["price"].agg(["mean"]),2).T
df = df.merge(z.T,how="left",on="car_company")
bins = [0,10000,20000,40000]
cars_bin=['Low','Medium','High']
df['CarsRange'] = pd.cut(df['mean'],bins,right=False,labels=cars_bin)
df = df.drop("car_company",axis=1)

df.dtypes

df["CarsRange"] = pd.Series(df["CarsRange"], dtype="object")
t = df.dtypes
for i in t[t=="object"].index:
    one_hot_encod = pd.get_dummies(df[i], prefix=i)#,drop_first=True
    df = pd.concat([df,one_hot_encod],axis=1)
    df.drop(i,axis=1,inplace=True)
Y = df["price"].copy()
df = df.drop(['price', "mean"],axis=1)

x_train,x_test , y_train,y_test = train_test_split(df,Y,test_size=0.08, shuffle=True)
x_train,x_val , y_train,y_val = train_test_split(x_train,y_train,test_size=0.1)

sc = MinMaxScaler()
x_train_s = sc.fit_transform(x_train)
x_val_s   = sc.transform(x_val)
x_test_s  = sc.transform(x_test)
x_train_s.shape

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=x_train_s.shape[-1]))

    for i in range(hp.Int("num_layers",min_value=2, max_value=5, step=1)):
        model.add(
            keras.layers.Dense(

                units=hp.Int(f"units{i}", min_value=64, max_value=512, step=64),
                activation=hp.Choice("activation", ["selu","relu"]),
                kernel_initializer="normal"
            )
        )
    model.add(keras.layers.Dense(1,kernel_initializer="normal"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=["mae",keras.metrics.RootMeanSquaredError()],
    )
    return model

import keras_tuner
tuner = keras_tuner.RandomSearch( build_model, objective='val_loss',max_trials=300, directory="/kaggle/working/keras_tuner_car_price_kaggle")
tuner.search_space_summary()

tuner.search(x_train_s, y_train, epochs=20, validation_data=(x_val_s, y_val))

models = tuner.get_best_models()
best_model = models[0]
best_model.summary()

best_model.get_config()

history=best_model.fit(x_train_s, y_train, epochs=100, validation_data=(x_val_s, y_val))

epochs = 100
print(f"Training MAE : {history.history['loss'][-1]}")
print(f"Validation MAE : {history.history['val_loss'][-1]}")
plt.clf()
fig = plt.figure()
fig.suptitle('Graph of training loss and validation loss')
plt.plot(range(epochs), history.history['loss'], color='b',label="Loss")
plt.plot(range(epochs), history.history['val_loss'], color='r',label="Val_Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

pred=best_model.predict(x_test_s)
R2_NN=r2_score(y_true=y_test,y_pred=pred)

MAE_NN = best_model.evaluate(x_test_s,y_test,verbose=0)[1]
RMSE_NN = best_model.evaluate(x_test_s,y_test,verbose=0)[2]

lm = LinearRegression()
lm.fit(x_train_s,y_train)
y_predicted = lm.predict(x_test_s)
R2_Score_MLR = lm.score(x_test_s,y_test)
MAE_MLR = mean_absolute_error(y_test, y_predicted)
RMSE_MLR = sqrt(mean_squared_error(y_test, y_predicted))

print(tabulate(pd.DataFrame(data={"R2 Score":[R2_NN*100,R2_Score_MLR*100],"RMSE":[RMSE_NN,RMSE_MLR],"MAE":[MAE_NN,MAE_MLR]},index=["Neural Network","Multi-Variable Regression"]), headers="keys", tablefmt='fancy_grid'))















