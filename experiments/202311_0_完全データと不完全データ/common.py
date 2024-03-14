import plotly.offline as pyoff
import plotly.graph_objs as go
from typing import List

import torch

from torch.utils.data import DataLoader, Dataset

from lpvis.common.types import Coords
from lpvis.visualizer.coords_visualizer import CoordsVisualizer
from lpvis.visualizer.time_series_visualizer import TimeSeriesVisualizer

import numpy as np
import copy
import polars as pl

class Model(torch.nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim):
        super().__init__()
        
        self.emb = torch.nn.Linear(in_dim, emb_dim, bias=True)
        self.head1 = torch.nn.Linear(emb_dim, out_dim, bias=True)
        self.head2 = torch.nn.Linear(emb_dim, out_dim, bias=True)
        
    def forward(self, x):
        z = self.emb(x)
        out1 = self.head1(z)
        out2 = self.head2(z)
 
        return [out1, out2]
    
    def params(self):
        _param_dict = {}
        
        for key in self.state_dict().keys():
            val = self.state_dict()[key].to('cpu').detach().numpy().flatten().tolist()
            
            _param_dict[key] = val
        return _param_dict


def plot_loss(train_loss_history, val_loss_history):
    trace0 = go.Scatter(x = [i for i in range(len(train_loss_history))], y =train_loss_history , mode='lines', name ='train')
    trace1 = go.Scatter(x = [i for i in range(len(val_loss_history))], y =val_loss_history , mode='lines', name ='val')

    layout = go.Layout(xaxis = dict(title="epochs"),   
                       yaxis = dict(title="loss"))

    fig = dict(data = [trace0, trace1], layout = layout)
    pyoff.iplot(fig)

def train_stl(train_X: Coords, val_X: Coords, model: torch.nn.Module, epochs: int = 10, model_save_step_interval: int = 200) -> List[torch.nn.Module]:
    models = []
    
    train_dataset = train_X.to_dataset()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = val_X.to_dataset()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs+1):
        train_total_loss = 0
        model.train()
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred[0], y_train.view(-1, 1).float())
            if epoch !=0:
                loss.backward()
                optimizer.step()
            train_total_loss += loss.item()

            if epoch!= 0 and step != 0 and step != len(train_loader) and step % model_save_step_interval == 0:
                models.append(copy.deepcopy(model))

        train_loss_history.append(train_total_loss)
        models.append(copy.deepcopy(model))

        # validation
        val_total_loss = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                loss = criterion(y_pred[0], y_val.view(-1, 1).float())
                val_total_loss += loss.item()
            
            val_loss_history.append(val_total_loss)

        if (epoch +1) % 1 == 0:
            print()
            print(f"epoch: {epoch+1}")
            print(f"train: {train_total_loss}")
            print(f"val  : {val_total_loss}")
            
    plot_loss(train_loss_history, val_loss_history)
    
    return models

def plot_data_with_line(X: Coords, model: torch.nn.Module, task_num: int = 0, x_range=[-16,16], y_range=[-16,16], resolution: int = 1000, dtick: int = 5):
    vis = CoordsVisualizer()

    # 特徴量をもとに新しいCoordsを作成
    dim_cols = [f"dim{i}" for i in range(X.dim)]
    dim0_list = []
    dim1_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_emb = copy.deepcopy(X)
    model = model.to(device)
    for i in range(len(X.df)):
        z = model.emb(torch.tensor(X.df[i].select(dim_cols).to_numpy()).float().to(device))
        dim0_list.append(z[0,0].item())
        dim1_list.append(z[0,1].item())

    # 特徴空間で描画
    emb_df = pl.DataFrame({"z_dim0": dim0_list, "z_dim1": dim1_list})
    X_emb.df = pl.concat([X_emb.df, emb_df], how="horizontal")
    X_emb.df = X_emb.df.drop(["dim0", "dim1"])
    X_emb.df = X_emb.df.rename({"z_dim0": "dim0", "z_dim1": "dim1"})
    fig_emb = vis.visualize_2d(X_emb)


    # X.dfの dim0, dim1 の列の最小値・最大値を取得
    x_min, x_max = x_range
    # y_min, y_max = y_range

    # x_list = np.linspace(x_min, x_max, resolution)
    # y_list = np.linspace(y_min, y_max, resolution)

    # z_mat = np.zeros((len(x_list), len(y_list)))
    # for i, x in enumerate(x_list):
    #     for j, y in enumerate(y_list):
    #         if task_num == 0:
    #             z_mat[i, j] = model.head1(torch.tensor([x, y]).float().to(device))[0].item()
    #         else:
    #             z_mat[i, j] = model.head2(torch.tensor([x, y]).float().to(device))[0].item()


    # 直線を引く
    # x座標の範囲を定義
    x = np.linspace(x_min, x_max, resolution)
    # 直線の式に従ってy座標を計算
    param_dict = model.params()
    # task1の場合
    weight = param_dict[f"head{task_num+1}.weight"]
    bias = param_dict[f"head{task_num+1}.bias"]
    y = -weight[0]/weight[1] * x - bias[0]/weight[1] 
    line = go.Scatter(x=x, y=y, mode='lines', name=f"y = {-weight[0]/weight[1]:.2f}x{'+' if -bias[0]/weight[1] > 0 else ''}{-bias[0]/weight[1]:.2f}")
    fig_emb.add_trace(line)


    fig_emb.update_layout(
        width=500, 
        height=500,
        xaxis = dict(range = x_range, dtick=dtick),   
        yaxis = dict(range = y_range, dtick=dtick, scaleanchor='x')
    )
    # 散布図を作成
    # contour = go.Contour(
    #     z=z_mat,
    #     x=x_list, # horizontal axis
    #     y=y_list, # vertical axis
    #     contours_coloring='lines',
    #     line_width=2,
    #     contours=dict(
    #         start=0,
    #         end=0,
    #         size=2,
    #     ),
    #     )
    # fig_emb.add_trace(contour)

    return fig_emb


def plot_data_animation_with_line(X: Coords, models: List[torch.nn.Module]):
    vis = TimeSeriesVisualizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 特徴量をもとに新しいCoordsを作成
    dim0_list = []
    dim1_list = []
    labels = []
    t_list = []
    for t, model in enumerate(models):
        model = model.to(device)
        model.eval()
        for x, y in X.to_dataset():
            z = model.emb(x.to(device))
            dim0_list.append(z[0].item())
            dim1_list.append(z[1].item())
            labels.append(y.item())
            t_list.append(t)

    ids = [i for i in range(len(dim0_list))]

    df = pl.DataFrame({"id": ids, "dim0": dim0_list, "dim1": dim1_list, "label": labels, "t": t_list})
    fig_emb = vis.visualize_2d(df)

    return fig_emb

@torch.no_grad()
def eval_stl(train_X: Coords, val_X: Coords, model: torch.nn.Module):
    for t, X in zip(["train", "val"],[train_X, val_X]):
        dataset = X.to_dataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        y_pred_list = []
        y_gt_list = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred_val = y_pred[0].detach().cpu().numpy().flatten()
            y_pred_np = np.array([1 if y_pred_val[0] > 0 else 0])
            y_gt_np = y.detach().cpu().numpy()
            y_pred_list.append(y_pred_np)
            y_gt_list.append(y_gt_np)

        y_pred = np.concatenate(y_pred_list)
        y_gt = np.concatenate(y_gt_list)
        accuracy = np.sum(y_pred == y_gt) / len(y_pred)
        print(f"{t} accuracy: {accuracy}")

class MTLDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.num_tasks = len(datasets)
        self.coords = datasets[0].coords
        self.labels_list = [dataset.labels for dataset in datasets]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        label_tensor = torch.tensor([labels[idx] for labels in self.labels_list], dtype=torch.long)
        return self.coords[idx], label_tensor


def train_mtl(train_X_list: List[Coords], val_X_list: List[Coords], model: torch.nn.Module, epochs: int = 10, model_save_step_interval: int = 200, lambda_list: List[float]=[0.5, 0.5]) -> List[torch.nn.Module]:
    models = []

    num_tasks = len(train_X_list)
    
    train_dataset = MTLDataset([train_X.to_dataset() for train_X in train_X_list])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = MTLDataset([val_X.to_dataset() for val_X in val_X_list])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs+1):
        train_total_loss = 0
        model.train()
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = lambda_list[0] * criterion(y_pred[0], y_train[0, 0].view(-1, 1).float())
            for i in range(1, num_tasks):
                loss += lambda_list[i] * criterion(y_pred[i], y_train[0, i].view(-1, 1).float())

            if epoch !=0:
                loss.backward()
                optimizer.step()
            train_total_loss += loss.item()

            if epoch!= 0 and step != 0 and step != len(train_loader) and step % model_save_step_interval == 0:
                models.append(copy.deepcopy(model))

        train_loss_history.append(train_total_loss)
        models.append(copy.deepcopy(model))

        # validation
        with torch.no_grad():
            val_total_loss = 0
            model.eval()
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_pred = model(x_val)
                loss = lambda_list[0] * criterion(y_pred[0], y_val[0, 0].view(-1, 1).float())
                # val は task 1 だけ
                # for i in range(1, num_tasks):
                #     loss += lambda_list[i] * criterion(y_pred[i], y_val[0, i].view(-1, 1).float())
                val_total_loss += loss.item()
            
            val_loss_history.append(val_total_loss)

        if (epoch +1) % 1 == 0:
            print()
            print(f"epoch: {epoch+1}")
            print(f"train: {train_total_loss}")
            print(f"val  : {val_total_loss}")
            
    plot_loss(train_loss_history, val_loss_history)
    
    return models

@torch.no_grad()
def eval_mtl(train_X_list: List[Coords], val_X_list: List[Coords], model: torch.nn.Module):

    train_dataset = MTLDataset([train_X.to_dataset() for train_X in train_X_list])
    val_dataset = MTLDataset([val_X.to_dataset() for val_X in val_X_list])

    for t, dataset in zip(["train", "val"],[train_dataset, val_dataset]):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        num_tasks = len(train_X_list)
        y_pred_list = [[] for _ in range(num_tasks)]
        y_gt_list = [[] for _ in range(num_tasks)]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        for x, y in dataloader:
            x = x.to(device)
            y_pred = model(x)

            for i in range(num_tasks):
                y_pred_i = y_pred[i].detach().cpu().numpy().flatten()
                y_pred_np = np.array([1 if y_pred_i > 0 else 0])
                y_gt_np = y[0,i].view(1).detach().cpu().numpy()
                y_pred_list[i].append(y_pred_np)
                y_gt_list[i].append(y_gt_np)


        for i in range(num_tasks):
            # task 2 以降についてはvalはいらない
            if t != "val" or i == 0:
                y_pred = np.concatenate(y_pred_list[i])
                y_gt = np.concatenate(y_gt_list[i])
                accuracy = np.sum(y_pred == y_gt) / len(y_pred)
                print(f"{t} task{i} accuracy: {accuracy}")
            
