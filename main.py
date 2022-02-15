import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# load data from Kaggle
cars = pd.read_csv('../input/carprice/CarPriceTest.csv')
cars.head(cars.shape[0])

#lets check if cars has unique names
print(cars['CarName'].unique())
cars.shape
cars.describe()
cars.info()

carNames = cars['CarName'].unique()
print("all car names: ", carNames)

#remove model name from the carName column
cars['CarName'] = cars['CarName'].apply(lambda x : x.split(' ')[0])

#rename column CarName to Brand
cars = cars.rename(columns={"CarName": "Brand"})

# Get the unique Brand names to make sure data is correct
print(cars['Brand'].unique())

cars = cars.replace({'porsche': 'porsche'})
cars = cars.replace({'porcshce': 'porsche'})
cars = cars.replace({'toyouta': 'toyota'})
cars = cars.replace({'vokswagen': 'volkswagen'})
cars = cars.replace({'vw': 'volkswagen'})
cars = cars.replace({'Nissan': 'nissan'})
cars = cars.replace({'maxda': 'mazda'})
cars = cars.replace({'alfa-romero': 'alfa romeo'})

print(cars['Brand'].unique())

cars = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower',
                'carlength','carwidth', 'citympg', 'highwaympg']]

# convert string values into numerical values
dummy = pd.get_dummies(cars[['fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber']])
cars = pd.concat([cars, dummy], axis=1)
cars.head(cars.shape[0])

#drop
cars = cars.drop(columns=['fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber'])
pd.options.display.max_columns = None
cars.head()

from sklearn.preprocessing import MinMaxScaler

# # price_column = cars.pop('price')
# # cars.insert(35, 'price', price_column)

#transform the data
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','citympg', 'highwaympg','carlength','carwidth','price']
cars[num_vars] = scaler.fit_transform(cars[num_vars])

pd.options.display.max_columns = None
cars.head(cars.shape[0])

plt.figure(figsize = (25, 25))
sns.heatmap(cars.corr(), cmap="YlGnBu", annot=True)
plt.show()

Y = cars.pop('price')
X = cars

from sklearn.model_selection import train_test_split
np.random.seed(0)
x_cars_train, x_cars_test, y_cars_train, y_cars_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=100)
x_cars_train

lr = LinearRegression()

lr.fit(x_cars_train,y_cars_train)

#Recursive Feature Elimination (RFE)
# Lets select top 10 features
rfe = RFE(lr, 10)
rfe = rfe.fit(x_cars_train, y_cars_train)

list(zip(x_cars_train.columns, rfe.support_, rfe.ranking_))

x_cars_train.columns[rfe.support_]

x_cars_train_rfe = x_cars_train[x_cars_train.columns[rfe.support_]]
x_cars_train_rfe.head()

model = sm.OLS(y_cars_train, sm.add_constant(x_cars_train_rfe)).fit()

X_cars_train_new = x_cars_train_rfe.drop(columns=['wheelbase', 'curbweight', 'highwaympg', 'carbody_convertible', 'enginetype_dohcv', 'enginetype_rotor', 'cylindernumber_eight', 'cylindernumber_twelve'])

x = sm.add_constant(X_cars_train_new)
model = sm.OLS(y_cars_train, x).fit()
model.summary()

lm = sm.OLS(y_cars_train,X_cars_train_new).fit()
y_train_price = lm.predict(X_cars_train_new)

X_test_new = sm.add_constant(x_cars_test_new)
lm1 = sm.OLS(y_cars_test, X_test_new).fit()
y_pred = lm1.predict(X_test_new)

from sklearn.metrics import r2_score 
r2_score = r2_score(y_cars_test, y_pred)
r2_score

