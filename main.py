import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('US_Accidents.csv', low_memory=False)
print(df.shape, df.dtypes)

# Drop high-missing columns
missing = df.isnull().mean()
df.drop(columns=missing[missing > 0.4].index, inplace=True)

# Fill numeric with median, categorical with mode
for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Feature engineering
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['hour']        = df['Start_Time'].dt.hour
df['day_of_week'] = df['Start_Time'].dt.dayofweek
df['month']       = df['Start_Time'].dt.month
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
df['is_night']    = df['hour'].between(20, 6).astype(int)

le = LabelEncoder()
for col in ['Weather_Condition', 'Wind_Direction', 'Sunrise_Sunset']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

df.to_csv('data/clean_accidents.csv', index=False)