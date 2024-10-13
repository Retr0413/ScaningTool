import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    x = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def save_processed_data(x_train, x_test, y_train, y_test, prefix):
    np.save(f'data/processed/{prefix}_x_train.npy', x_train)
    np.save(f'data/processed/{prefix}_x_test.npy', x_test)
    np.save(f'data/processed/{prefix}_y_train.npy', y_train)
    np.save(f'data/processed/{prefix}_y_test.npy', y_test)

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_data('data/datasets/cve.csv')
    save_processed_data('vuln', x_train, x_test, y_train, y_test)

    x_train, x_test, y_train, y_test = preprocess_data('data/datasets/csic_database.csv')
    save_processed_data('ddos', x_train, x_test, y_train, y_test)