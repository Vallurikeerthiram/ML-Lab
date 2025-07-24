import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(file_path, features, threshold=25):
    df = pd.read_csv(file_path, encoding='latin1')
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=features, inplace=True)
    df['Class'] = (df['Pre-monsoon_2022 (meters below ground level)'] >= threshold).astype(int)
    return df

def save_split_data(df, features):
    train_df, test_df = train_test_split(df[features + ['Class']], test_size=0.3, random_state=42)
    train_df.to_csv('groundwater_train.csv', index=False)
    test_df.to_csv('groundwater_test.csv', index=False)
    return train_df, test_df

# Main
features = [
    'Pre-monsoon_2015 (meters below ground level)',
    'Pre-monsoon_2016 (meters below ground level)',
    'Pre-monsoon_2017 (meters below ground level)',
    'Pre-monsoon_2018 (meters below ground level)',
    'Pre-monsoon_2019 (meters below ground level)',
    'Pre-monsoon_2020 (meters below ground level)',
    'Pre-monsoon_2021 (meters below ground level)',
    'Pre-monsoon_2022 (meters below ground level)'
]
df = prepare_data("Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv", features)
train_df, test_df = save_split_data(df, features)

print("Training and testing data saved.")
