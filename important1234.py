import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Define file paths
train_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Training.csv')
test_file_path = os.path.join('/kaggle/input/poljugy', 'Renewable_Testing.csv')

# Check if the files exist
if os.path.exists(train_file_path):
    print(f"Training file found at: {train_file_path}")
else:
    print(f"Training file not found at: {train_file_path}")

if os.path.exists(test_file_path):
    print(f"Testing file found at: {test_file_path}")
else:
    print(f"Testing file not found at: {test_file_path}")

# Only proceed if both files exist
if os.path.exists(train_file_path) and os.path.exists(test_file_path):
    # Load the training and testing datasets
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

else:
    print("One or both CSV files are missing. Please check the file paths.")
