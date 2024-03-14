
# IMPORT
import sys
sys.path.append("/workspace")
sys.path.append("/workspace/experiments/202312")

from torchvision.datasets import CelebA
from torchvision import transforms
from typing import List
import copy
from common import plot_loss
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import pandas as pd
import time
import datetime

# CONST
EXP_ID = 20 # 実験番号
N = 10
MAX_EPOCHS = 30
DATA_SIZE = 1000
TASK_NUM = 40

# test
# N = 3
# MAX_EPOCHS = 1
# DATA_SIZE = 100
# TASK_NUM = 3

ALL_TASK_IDS = [i for i in range(TASK_NUM)]



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_dataset = CelebA(root='/workspace/data', split='train', download=True, transform=transform)
val_dataset = CelebA(root='/workspace/data', split='valid', download=True, transform=transform)
test_dataset = CelebA(root='/workspace/data', split='test', download=True, transform=transform)

subset_indices = list(range(DATA_SIZE))  # 最初の1000個のインデックス
train_subset = Subset(train_dataset, subset_indices)
val_subset = Subset(val_dataset, subset_indices)
test_subset = Subset(test_dataset, subset_indices)

train_dl = DataLoader(train_subset, batch_size=32, num_workers=4, shuffle=True) 
val_dl = DataLoader(val_subset, batch_size=32, num_workers=4, shuffle=False)
test_dl = DataLoader(test_subset, batch_size=32, num_workers=4, shuffle=False)

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


def train_stl(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, task_id: int = 0, epochs: int = 10, model_save_step_interval: int = 10, verbose=True):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loss_history = []
    val_loss_history = []

    # models = []

    for epoch in range(epochs+1):
        model.train()
        train_total_loss = 0
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(x_train)

            task1_gt = y_train[:, task_id].view(-1, 1).float() 
            loss = criterion(y_pred[0], task1_gt)

            if epoch !=0:
                loss.backward()
                optimizer.step()
            train_total_loss += loss.item()

            # if epoch!= 0 and step != 0 and step != len(train_loader) and step % model_save_step_interval == 0:
                # models.append(copy.deepcopy(model))

        train_loss_history.append(train_total_loss)
        # models.append(copy.deepcopy(model))

        # validation
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            for step, (x_val, y_val) in enumerate(val_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                task1_gt = y_val[:, task_id].view(-1, 1).float() 
                loss = criterion(y_pred[0], task1_gt) 
                val_total_loss += loss.item()
            
            val_loss_history.append(val_total_loss)

        if (epoch +1) % 1 == 0 and verbose:
            print()
            print(f"epoch: {epoch+1}")
            print(f"train: {train_total_loss}")
            print(f"val  : {val_total_loss}")
            
    if verbose:
        plot_loss(train_loss_history, val_loss_history)
    
    # return models
    return model

# 不完全データとして学習を行う
def train_mtl(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, task_ids: List[int] = [0,1], epochs: int = 10, model_save_step_interval: int = 10, lambda_list: List[float]=[0.5, 0.5], verbose=True):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loss_history = []
    val_loss_history = []

    # models = []

    for epoch in range(epochs+1):
        model.train()
        train_total_loss = 0
        for step, (x_train, y_train, task_labels) in enumerate(train_loader):
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
        # models.append(copy.deepcopy(model))

        # validation
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            for step, (x_val, y_val) in enumerate(val_loader):
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
    
    # return models
    return model

@torch.no_grad()
def eval(model: torch.nn.Module, val_dl: DataLoader, target_task_id = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_pred_list = []
    y_gt_list = []

    for step, (x, y) in enumerate(val_dl):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        y_pred_i = y_pred[0].cpu().detach().numpy().astype(int)
        y_pred_i[y_pred_i > 0] = 1
        y_pred_i[y_pred_i <= 0] = 0
        y_pred_i = y_pred_i.flatten().tolist()
        y_gt = y[:, target_task_id].cpu().detach().numpy().flatten().tolist()
        y_pred_list.extend(y_pred_i)
        y_gt_list.extend(y_gt)


    y_pred = np.array(y_pred_list)
    y_gt = np.array(y_gt_list)
    accuracy = np.sum(y_pred == y_gt) / len(y_pred)

    return accuracy

class EndTimePredictor:
    def __init__(self, start_time, total_iterations):
        self.start_time = start_time
        self.total_iterations = total_iterations
        self.offset = datetime.timedelta(hours=9)
        self.current_iteration = 0

    def show_predicted_end_time(self):
        self.current_iteration += 1
        
        # 現在のイテレーションまでの平均時間を計算
        elapsed_time = time.time() - self.start_time
        average_time_per_iteration = elapsed_time / self.current_iteration
        # 残りのイテレーションにかかる予測時間を計算
        remaining_time = average_time_per_iteration * (self.total_iterations - self.current_iteration)
        # 終了予測時刻を計算
        predicted_end_time_utc = datetime.datetime.fromtimestamp(self.start_time + elapsed_time + remaining_time)
        predicted_end_time = predicted_end_time_utc + self.offset
        # 終了予測時刻を表示
        print(f"current iteration: {self.current_iteration}/{self.total_iterations}")
        print("Predicted end time:", predicted_end_time.strftime('%Y-%m-%d %H:%M:%S'))

start_time = time.time()  # 開始時刻
total_iterations = N * len(ALL_TASK_IDS) * len(ALL_TASK_IDS) # 総イテレーション数
end_time_predictor = EndTimePredictor(start_time, total_iterations)

for target_task_id in ALL_TASK_IDS:
    print(f"target_task_id: {target_task_id}, target_task_name: {attr_names[target_task_id]}")

    acc_list_dict = {}
    source_task_ids = [i for i in ALL_TASK_IDS if i != target_task_id]

    # STL
    print("STL")
    stl_acc_list = []
    for i in range(N):
        print(f"iter: {i+1}/{N}")
        stl_model = SimpleCNN()
        stl_model = train_stl(stl_model, train_dl, val_dl, task_id=target_task_id, epochs=MAX_EPOCHS, verbose=False)
        target_task_acc = eval(stl_model, val_dl, target_task_id=target_task_id)
        stl_acc_list.append(target_task_acc)

        # 終了予測時刻を表示
        end_time_predictor.show_predicted_end_time()

    acc_list_dict[str(target_task_id)] = stl_acc_list

    # MTL
    print("MTL")
    for aux_task_id in source_task_ids:
        print(f"source_task_id: {aux_task_id}, source_task_name: {attr_names[aux_task_id]}")
        
        # 繰り返し実験
        mtl_acc_list = []

        for i in range(N):
            print(f"iter: {i+1}/{N}")
            # Dataset を作成（データの不完全化）
            task_labels = np.random.randint(0, 2, size=len(train_dl.dataset)).tolist() 
            custom_dataset = DatasetForMTL(train_dl.dataset, task_labels)
            mtl_train_dl = DataLoader(custom_dataset, batch_size=32, num_workers=4, shuffle=True)

            mtl_model = SimpleCNN()
            mtl_model = train_mtl(mtl_model, mtl_train_dl, val_dl, task_ids=[target_task_id, aux_task_id], epochs=MAX_EPOCHS, lambda_list=[0.5, 0.5], verbose=False)
            target_task_acc = eval(mtl_model, val_dl, target_task_id=target_task_id)
            mtl_acc_list.append(target_task_acc)

            # 終了予測時刻を表示
            end_time_predictor.show_predicted_end_time()

        acc_list_dict[str(aux_task_id)] = mtl_acc_list

    # polars df に変換
    df = pl.DataFrame(acc_list_dict)
    print(df)

    print(df)
    csv_name = f"./csv/exp{EXP_ID}_target_task={target_task_id}_iter={N}.csv"
    df.write_csv(csv_name)

    data = pd.read_csv(csv_name)
    data = data.rename(columns={f'{target_task_id}': 'STL'})

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
    ax.set_title(f'Violin Plots: target_task={attr_names[target_task_id]}')

    plt.savefig(f"./images/exp{EXP_ID}_target_task={target_task_id}_iter={N}.png")
    plt.show()
