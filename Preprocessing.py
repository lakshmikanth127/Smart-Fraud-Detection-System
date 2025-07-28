import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
