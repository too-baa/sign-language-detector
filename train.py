import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load the data
if not os.path.exists('dataset.csv'):
    print("Error: dataset.csv not found!")
else:
    # We read without a header and define our own column names
    # Column 0 is the 'label', Columns 1-42 are the landmark coordinates
    df = pd.read_csv('dataset.csv', header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    
    X = df.drop('label', axis=1) # Coordinates
    y = df['label']              # Letters
    
    # 2. Train the model
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # 3. Save the model
    joblib.dump(model, 'sign_model.pkl')
    print("Success! Model trained and saved as sign_model.pkl!")