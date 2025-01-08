#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load the training and testing datasets
train_file_path = os.path.join('/kaggle/input/renewable-training-csv', 'Renewable_Training.csv')
test_file_path = os.path.join('/kaggle/input/renewable-testing-csv', 'Renewable_Testing.csv')

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Separate features and target variable
X_train = train_data.drop(['Energy delta[Wh]'], axis=1)
y_train = train_data['Energy delta[Wh]']

X_test = test_data.drop(['Energy delta[Wh]'], axis=1)
y_test = test_data['Energy delta[Wh]']

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the trained model
model_filename = 'predictive_load_model.pkl'
joblib.dump(model, model_filename)
print(f'Model saved as {model_filename}')


# In[3]:


import os
for root, dirs, files in os.walk('/kaggle/input/'):
    print(f'Root: {root}')
    print(f'Dirs: {dirs}')
    print(f'Files: {files}')



# In[7]:


train_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Training.csv')
test_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Testing.csv')


# In[9]:


# Load the training and testing datasets
train_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Training.csv')
test_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Testing.csv')

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)


# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load the training and testing datasets
train_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Training.csv')
test_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Testing.csv')

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Separate features and target variable
X_train = train_data.drop(['Energy delta[Wh]'], axis=1)
y_train = train_data['Energy delta[Wh]']

X_test = test_data.drop(['Energy delta[Wh]'], axis=1)
y_test = test_data['Energy delta[Wh]']

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Save the trained model
model_filename = 'predictive_load_model.pkl'
joblib.dump(model, model_filename)
print(f'Model saved as {model_filename}')


# In[12]:


feature_importances = pd.DataFrame(
    model.feature_importances_, 
    index=X_train.columns, 
    columns=["Importance"]
).sort_values(by="Importance", ascending=False)
print(feature_importances)


# In[13]:


feature_filename = 'model_features.pkl'
joblib.dump(X_train.columns, feature_filename)
print(f'Feature names saved as {feature_filename}')


# In[14]:


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Energy delta [Wh]')
plt.ylabel('Predicted Energy delta [Wh]')
plt.title('Actual vs. Predicted')
plt.show()


# In[15]:


print(train_data.isnull().sum())
print(test_data.isnull().sum())

# Example handling
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)


# In[16]:


import seaborn as sns

for column in X_train.columns:
    sns.boxplot(x=train_data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()


# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Example models
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel='rbf'),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'{name} - MSE: {mse}, R^2: {r2}')

