import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.vulnerability_model import VulnerabilityModel  # 脆弱性モデルのインポート

# GPUが利用可能ならGPUを使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_test_data(prefix):
    """テストデータを読み込む関数"""
    X_test = np.load(f'data/processed/{prefix}_X_test.npy')
    y_test = np.load(f'data/processed/{prefix}_y_test.npy')
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    """モデルの評価と結果の可視化"""
    # データをTensorに変換
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # モデルを評価モードに設定し、予測を実行
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().round()

    # 評価レポートを出力
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # 混同行列の生成と表示
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # 脆弱性モデルのロードと評価
    X_test, y_test = load_test_data('vuln')

    # モデルの構築とロード
    vuln_model = VulnerabilityModel(X_test.shape[1]).to(device)
    vuln_model.load_state_dict(torch.load('models/vulnerability_model.pth', map_location=device))

    print("Evaluating Vulnerability Model...")
    evaluate_model(vuln_model, X_test, y_test)
