import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def preprocess_cve_data(file_path):
    """
    cve.csv データの前処理。
    脆弱性のスコアや影響に関する特徴量を処理。
    """
    df = pd.read_csv(file_path)

    # 各特徴量の取得
    cvss = df['cvss'].values.reshape(-1, 1)  # CVSSスコア
    cwe_code = df['cwe_code'].values.reshape(-1, 1)  # CWEコード

    # カテゴリ特徴量のラベルエンコード
    encoder = LabelEncoder()
    access_authentication = encoder.fit_transform(df['access_authentication']).reshape(-1, 1)
    access_complexity = encoder.fit_transform(df['access_complexity']).reshape(-1, 1)
    access_vector = encoder.fit_transform(df['access_vector']).reshape(-1, 1)

    # 影響に関する特徴量
    impact_availability = df['impact_availability'].values.reshape(-1, 1)
    impact_confidentiality = df['impact_confidentiality'].values.reshape(-1, 1)
    impact_integrity = df['impact_integrity'].values.reshape(-1, 1)

    # 特徴量の結合
    X = np.hstack([
        cvss, cwe_code, access_authentication, access_complexity,
        access_vector, impact_availability, impact_confidentiality, impact_integrity
    ])

    # ラベル列のエンコード（summary列をカテゴリとして扱う）
    y = encoder.fit_transform(df['summary'])

    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def preprocess_csic_data(file_path):
    """
    csic_database.csv データの前処理。
    HTTPリクエストの詳細に関する特徴量を処理。
    """
    df = pd.read_csv(file_path)

    # HTTPメソッドのラベルエンコード
    encoder = LabelEncoder()
    method = encoder.fit_transform(df['Method']).reshape(-1, 1)

    # コンテンツタイプ、接続情報のラベルエンコード
    content_type = encoder.fit_transform(df['content-type']).reshape(-1, 1)
    connection = encoder.fit_transform(df['connection']).reshape(-1, 1)

    # 数値特徴量の取得
    lenght = df['lenght'].values.reshape(-1, 1)  # リクエストの長さ
    content = df['content'].values.reshape(-1, 1)  # コンテンツサイズ

    # 特徴量の結合
    X = np.hstack([method, content_type, connection, lenght, content])

    # ラベル列の取得（classification列をエンコード）
    y = encoder.fit_transform(df['classification'])

    # データの標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# データをNumpy形式で保存する関数
def save_processed_data(prefix, X_train, X_test, y_train, y_test):
    np.save(f'data/processed/{prefix}_X_train.npy', X_train)
    np.save(f'data/processed/{prefix}_X_test.npy', X_test)
    np.save(f'data/processed/{prefix}_y_train.npy', y_train)
    np.save(f'data/processed/{prefix}_y_test.npy', y_test)

if __name__ == '__main__':
    # cve.csv の前処理と保存
    print("Processing cve.csv...")
    X_train, X_test, y_train, y_test = preprocess_cve_data(
        'C:\\Users\\user\\Desktop\\ゼミ作成\\ScaningTool\\data\\datasets\\cve.csv'
    )
    save_processed_data('vuln', X_train, X_test, y_train, y_test)

    # csic_database.csv の前処理と保存
    print("Processing csic_database.csv...")
    X_train, X_test, y_train, y_test = preprocess_csic_data(
        'C:\\Users\\user\\Desktop\\ゼミ作成\\ScaningTool\\data\\datasets\\csic_database.csv'
    )
    save_processed_data('ddos', X_train, X_test, y_train, y_test)
