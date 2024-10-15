import torch
from torch.utils.data import DataLoader, TensorDataset
from models.ddos_model import DDOSModel
import numpy as np

# GPUの使用確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データの読み込み
x_train = np.load('data/processed/ddos_x_train.npy')
y_train = np.load('data/processed/ddos_y_train.npy')

# Tensorに変換
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoaderの作成
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# モデルの構築
input_size = x_train.shape[1]
model = DDOSModel(input_size).to(device)

# 損失関数と最適化手法
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(10):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# モデルの保存
torch.save(model.state_dict(), 'models/ddos_model.pth')
