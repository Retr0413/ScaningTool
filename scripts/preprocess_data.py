import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import os

def preprocess_cve_data(file_path):
    """
    cve.csv データの前処理。
    カテゴリ列をエンコードし、数値列は標準化。
    """
    df = pd.read_csv(file_path)

    # --- 数値特徴量の取得 ---
    numeric_features = ['cvss', 'cwe_code', 'impact_availability', 
                        'impact_confidentiality', 'impact_integrity']

    # 数値列の変換：文字列を NaN にし、欠損値を平均で埋める
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

    x_numeric = df[numeric_features].values

    # --- カテゴリ特徴量のエンコード ---
    categorical_features = ['access_authentication', 'access_complexity', 'access_vector']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_categorical = encoder.fit_transform(df[categorical_features])

    # --- 特徴量の結合 ---
    x = np.hstack([x_numeric, x_categorical])

    # --- ラベル列のエンコード（summary列をラベル化） ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['summary'])

    # --- 数値列の標準化 ---
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 訓練データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def preprocess_csic_data(file_path):
    """
    csic_database.csv データの前処理。
    HTTPリクエストの詳細をエンコードし、数値列は標準化。
    """
    df = pd.read_csv(file_path)

    # --- 数値特徴量の取得 ---
    numeric_features = ['lenght', 'content']
    
    # 数値列の変換：文字列を NaN にし、欠損値を平均で埋める
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

    x_numeric = df[numeric_features].values

    # --- カテゴリ特徴量のエンコード ---
    categorical_features = ['Method', 'content-type', 'connection']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_categorical = encoder.fit_transform(df[categorical_features])

    # --- 特徴量の結合 ---
    x = np.hstack([x_numeric, x_categorical])

    # --- ラベル列のエンコード（classification列） ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['classification'])

    # --- 数値列の標準化 ---
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # 訓練データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def create_directory_if_not_exists(directory):
    """ディレクトリが存在しない場合に作成します。"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# データをCSV形式で保存する関数
def save_processed_data(prefix, x_train, x_test, y_train, y_test):
    """CSVファイルを保存します。"""
    directory = 'data/processed'
    create_directory_if_not_exists(directory)  # ディレクトリが存在しなければ作成

    # データをCSV形式で保存
    pd.DataFrame(x_train).to_csv(f'{directory}/{prefix}_x_train.csv', index=False)
    pd.DataFrame(x_test).to_csv(f'{directory}/{prefix}_x_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{directory}/{prefix}_y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{directory}/{prefix}_y_test.csv', index=False)

if __name__ == '__main__':
    # cve.csv の前処理と保存
    print("Processing cve.csv...")
    x_train, x_test, y_train, y_test = preprocess_cve_data(
        '/Users/tarima/Desktop/Litalico/ScaningTool/data/datasets/cve.csv'
    )
    save_processed_data('vuln', x_train, x_test, y_train, y_test)

    # csic_database.csv の前処理と保存
    print("Processing csic_database.csv...")
    x_train, x_test, y_train, y_test = preprocess_csic_data(
        '/Users/tarima/Desktop/Litalico/ScaningTool/data/datasets/csic_database.csv'
    )
    save_processed_data('ddos', x_train, x_test, y_train, y_test)
