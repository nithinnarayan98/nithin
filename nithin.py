import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/nithi/Desktop/narayan.csv')
df.info()

#checking for null values
df.isnull().sum()

#CHecking for duplicate
df.duplicated().sum()

#droping duplicate
df=df.drop_duplicates()

#Checking for unique values in the names column
list(df['name'].unique())

#extracting brand name from name column as to minimize the qnique value count
df['brand_name']=df['name'].str.split().str[0]
df['brand_name']

df['brand_name'] = df['brand_name'].replace('Land', 'Land Rover')

#creating a new df with select columns
df1= df.drop(['fuel', 'seller_type', 'name'], axis = 1)
df1

#One-Hot key labeling
df1 = pd.get_dummies(df1, columns=['transmission', 'brand_name'], drop_first=True)

print(df1)

#label encoding owner column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1['owner'] = le.fit_transform(df1['owner'])

#adding new column
current_year = 2025
df1['car_age'] = current_year - df1['year']
df1.drop('year', axis=1, inplace=True)

#scalling the km driven column
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df1[['km_driven']] = scaler.fit_transform(df1[['km_driven']])

#checking for ouliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df1['selling_price'])
plt.title("Boxplot of Selling Price (Before Log Transform)")
plt.show()

# Apply log transformation to selling price to deal with the outliers
df1['selling_price'] = np.log1p(df1['selling_price'])

#checking for ouliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df1['selling_price'])
plt.title("Boxplot of Selling Price (Before Log Transform)")
plt.show()

#checking for ouliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df1['km_driven'])
plt.title("Boxplot of Selling Price (Before Log Transform)")
plt.show()

# Apply log transformation to km_driven to deal with the outliers
df1['km_driven'] = np.log1p(df1['km_driven'])

#checking for ouliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df1['km_driven'])
plt.title("Boxplot of Selling Price (Before Log Transform)")
plt.show()
df1['km_driven'].fillna(df1['km_driven'].median(), inplace=True)

#setting up x and y for model training and testing
x = df1.drop('selling_price', axis = 1)
y = df1['selling_price']
from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize and train the Linear Regression model
m = LinearRegression()
m.fit(x_train, y_train)

# Make predictions on the test set
y_pred = m.predict(x_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

import pickle

columns = x_train.columns
pickle.dump(columns, open('m-columns.pkl', 'wb'))
pickle.dump(m, open('m-lr.pkl', 'wb'))