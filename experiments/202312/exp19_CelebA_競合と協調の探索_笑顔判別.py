
# IMPORT
import sys
sys.path.append("/workspace")
sys.path.append("/workspace/experiments/202312")

from torchvision.datasets import CelebA
from torchvision import transforms
from typing import List
import copy
from common import plot_loss
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import pandas as pd

# CONST
MALE_TASK_ID = 20
MUSTACHE_TASK_ID = 22
BROWN_HAIR_TASK_ID = 11
SMILING_TASK_ID = 31
YOUNG_TASK_ID = 39
BAGS_UNDER_EYES_TASK_ID = 3

TARGET_TASK_ID = SMILING_TASK_ID
SOURCE_TASKS = [ i for i in range(40) if i != TARGET_TASK_ID]

N = 10
MAX_EPOCHS = 30

# N = 1
# MAX_EPOCHS = 1

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_dataset = CelebA(root='/workspace/data', split='train', download=True, transform=transform)
val_dataset = CelebA(root='/workspace/data', split='valid', download=True, transform=transform)
test_dataset = CelebA(root='/workspace/data', split='test', download=True, transform=transform)

train_dl = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False)
val_dl = DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=False)
test_dl = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

attr_names = train_dataset.attr_names # タスク名のdict

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1つ目の畳み込み層
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2つ目の畳み込み層
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全結合層
        self.fc1 = nn.Linear(16 * 16 * 16, 10) 
        self.relu3 = nn.ReLU()

        self.head1 = nn.Linear(10, 1)  # 2値分類なので出力は1クラス
        self.head2 = nn.Linear(10, 1)  # 2値分類なので出力は1クラス

    def emb(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 16 * 16)  # ベクトル化
        x = self.fc1(x)
        x = self.relu3(x)
        return x

    def forward(self, x):
        x = self.emb(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return [x1, x2]

class DatasetForMTL(Dataset):
    def __init__(self, original_dataset, task_labels):
        self.original_dataset = original_dataset
        self.task_labels = task_labels

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data, target = self.original_dataset[idx]
        task_label = self.task_labels[idx]
        return data, target, task_label


# 不完全データとして学習を行う
def train2(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, task_ids: List[int] = [0,1], epochs: int = 10, model_save_step_interval: int = 10, lambda_list: List[float]=[0.5, 0.5], verbose=True):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.train()

    train_loss_history = []
    val_loss_history = []

    models = []

    for epoch in range(epochs+1):
        model.train()
        train_total_loss = 0
        for step, (x_train, y_train, task_labels) in enumerate(train_loader):
            # 100ステップまでで止める
            if step == 100:
                break
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(x_train)


            # バッチ内の各データポイントの損失を計算するための変数
            loss = 0

            for i in range(x_train.size(0)):  # バッチサイズに応じてループ
                task_label = task_labels[i]
                task_gt = y_train[i, task_ids[task_label]].view(-1, 1).float()
                loss += lambda_list[task_label] * criterion(y_pred[task_label][i].unsqueeze(0), task_gt)

            if epoch !=0:
                loss.backward()
                optimizer.step()
            train_total_loss += loss.item()

            # if epoch!= 0 and step != 0 and step != len(train_loader) and step % model_save_step_interval == 0:
                # models.append(copy.deepcopy(model))

        train_loss_history.append(train_total_loss)
        models.append(copy.deepcopy(model))

        # validation
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            for step, (x_val, y_val) in enumerate(val_loader):
                # 100ステップまでで止める
                if step == 100:
                    break
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                task1_gt = y_val[:, task_ids[0]].view(-1, 1).float() 
                # task2_gt = y_val[:, task_ids[1]].view(-1, 1).float()
                loss = criterion(y_pred[0], task1_gt) 
                # val は task1 だけ
                #+ lambda_list[1] * criterion(y_pred[1], task2_gt)
                val_total_loss += loss.item()
            
            val_loss_history.append(val_total_loss)

        if (epoch +1) % 1 == 0 and verbose:
            print()
            print(f"epoch: {epoch+1}")
            print(f"train: {train_total_loss}")
            print(f"val  : {val_total_loss}")
            
    if verbose:
        plot_loss(train_loss_history, val_loss_history)
    
    return models

@torch.no_grad()
def eval2(model: torch.nn.Module, train_dl: DataLoader, val_dl: DataLoader, task_ids: List[int] = [0,1]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for t, dataloader in zip(["val"],[val_dl]):
        num_tasks = 2
        y_pred_list = [[] for _ in range(num_tasks)]
        y_gt_list = [[] for _ in range(num_tasks)]


        for step, (x, y) in enumerate(dataloader):
            # 100ステップまでで止める
            if step == 100:
                break
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            for i in range(num_tasks):
                y_pred_i = y_pred[i].cpu().detach().numpy().astype(int)
                y_pred_i[y_pred_i > 0] = 1
                y_pred_i[y_pred_i <= 0] = 0
                y_pred_i = y_pred_i.flatten().tolist()
                y_gt = y[:, task_ids[i]].cpu().detach().numpy().flatten().tolist()
                y_pred_list[i].extend(y_pred_i)
                y_gt_list[i].extend(y_gt)

        for i in range(num_tasks):
            # task 2 以降についてはvalはいらない
            if t != "val" or i == 0:
                y_pred = np.array(y_pred_list[i])
                y_gt = np.array(y_gt_list[i])
                accuracy = np.sum(y_pred == y_gt) / len(y_pred)
                print(f"{t} task{i} accuracy: {accuracy}")
            if t == "val" and i == 0:
                metrics = accuracy
    return metrics

mtl_acc_list_dict = {}

for aux_task_id in SOURCE_TASKS:
    print(f"source_task_id: {aux_task_id}, source_task_name: {attr_names[aux_task_id]}")
    
    # 繰り返し実験
    mtl_acc_list = []

    for i in range(N):
        print(f"iter: {i+1}/{N}")
        # Dataset を作成（データの不完全化）
        task_labels = np.random.randint(0, 2, size=len(train_dl.dataset)).tolist() 
        custom_dataset = DatasetForMTL(train_dl.dataset, task_labels)
        train_dl2 = DataLoader(custom_dataset, batch_size=32, num_workers=4, shuffle=False)

        mtl_model2 = SimpleCNN()
        mtl_models2 = train2(mtl_model2, train_dl2, val_dl, task_ids=[TARGET_TASK_ID, aux_task_id], epochs=MAX_EPOCHS, lambda_list=[0.5, 0.5], verbose=False)
        target_task_acc = eval2(mtl_model2, train_dl2, val_dl, task_ids=[TARGET_TASK_ID, aux_task_id])
        mtl_acc_list.append(target_task_acc)
        # print(f"target_task_acc: {target_task_acc}, target_task_id: {target_task_id}, target_task_name: {attr_names[target_task_id]}, aux_task_id: {aux_task_id}, aux_task_name: {attr_names[aux_task_id]}")

    mtl_acc_list_dict[attr_names[aux_task_id]] = mtl_acc_list

# polars df に変換
df = pl.DataFrame(mtl_acc_list_dict)
print(df)

print(df)
csv_name = f"./exp19_df_{N}.csv"
df.write_csv(csv_name)


data = pd.read_csv(csv_name)
# data = data.rename(columns={'stl': 'STL', 'mtl': 'MTL1', 'mtl2': 'MTL2'})

# データを長い形式に変換
melted_data = data.melt(var_name='Variable', value_name='Accuracy')

# プロットの初期化
fig, ax = plt.subplots(figsize=(10, 6))

# 各変数に対してプロットを作成
for col in melted_data['Variable'].unique():
    subset = melted_data[melted_data['Variable'] == col]
    # バイオリンプロットの半分
    sns.violinplot(x='Variable', y='Accuracy', data=subset, fill=False, split=False, ax=ax)

# タイトルの設定
ax.set_title('Violin Plots')

plt.savefig(f"./images/violin_{N}.png")
plt.show()
