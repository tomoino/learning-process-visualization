from typing import Tuple

import numpy as np
import polars as pl
import torch

from sklearn.utils import check_random_state

from lpvis.dim_reducer.basic_dim_reducer import BasicDimReducer

import random
import math
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state

floath = np.float32

def prob_one(prob):
    return 1 if random.random() < prob else 0

def make_B(labels):
    num_samples = len(labels)
    unique_labels = sorted(list(set(labels)))
    num_unique_labels = len(unique_labels)
    num_references = num_unique_labels
    B = np.zeros((num_samples, num_references))
    num_references_for_each_label = num_references//num_unique_labels
    
    PROB_A = 0 # 相関のない reference を 1 にする確率 
    PROB_B = 1 # 相関のある reference を 1 にする確率 
    
    for row_num, label in enumerate(labels):
        label_id = unique_labels.index(label)
        
        # 相関のない reference を確率的に 1 にする
        B[row_num, :] = [ prob_one(PROB_A) for i in range(num_references)]
        
        for col_num in range(num_references):
            if math.floor(col_num/num_references_for_each_label) == label_id:
                B[row_num, col_num] = prob_one(PROB_B)
    
    return B

def make_inputs_for_log_bilinear(coords_df, labels, t_column="t"):

    list_Z = []
    max_step = coords_df[t_column].max()

    for step in range(max_step+1):
        step_df = coords_df.filter(pl.col(t_column) == step).drop(t_column)
        list_Z.append(step_df)
        if step == max_step:
            N = len(step_df)
            B = make_B(labels[N*step:N*(step+1)])

    list_Z.reverse()

    return B, list_Z, N

def calc_R(X, Y):
    R = np.array(X) @ np.array(Y).T
    return R

def calc_S(Z, C):    
    S = np.array(Z) @ np.array(Z).T * C
    return S

def calc_RR(R):
    RR = np.array(R) @ np.array(R).T
    return RR

def make_list_S(list_Z):
    """Sのリスト作成"""
    list_S = []
    
    # 類似度の計算のためにZの絶対値最大を調べる
    max_norm = 0
    for Z in list_Z:
        norms = np.linalg.norm(Z, axis=1)
        _max_norm = np.max(norms)
        if _max_norm > max_norm:
            max_norm = _max_norm
    C = (1.0/max_norm)**2
    
    for Z in list_Z:
        Z_mat = Z.to_numpy()
        list_S.append(C*Z_mat@Z_mat.T)
    
    return list_S, C

def prob_mat(mat):
    if isinstance(mat, np.ndarray):
        return normalize(np.exp(mat), norm='l1', axis=1)
    else: # torch tensor
        res_mat = torch.nn.functional.normalize(torch.exp(mat), p=1, dim=1)
        if not torch.allclose(res_mat.sum(dim=1), torch.ones(res_mat.shape[0]).to(res_mat.device)):
            print("prob_mat failed")

        return res_mat

    
def prob_mat_list(mats):
    """
    list に格納された行列をそれぞれ行ごとに確率化する
    """
    probs = []
    for mat in mats:
        probs.append(prob_mat(mat))
    
    return probs


def calc_KL(P, Q) -> float:
    """KLダイバージェンスを計算する

    Parameters
    ----------
    P : np.ndarray
        観測データから計算される確率分布
    Q : np.ndarray
        推定された確率分布

    Returns
    -------
    float
        KLダイバージェンスの値
    """
    if isinstance(P, np.ndarray):
        kl = np.sum(P * np.log(P / (Q + 1e-12)))
    else: # torch
        kl = torch.sum(P * torch.log(P / (Q + 1e-12)))
    return kl

def calc_loss_at_t(t, P_S, P_B, X, X_prev, Y, lambda1, lambda2):
    
    R = X @ Y.T
    Q_R = prob_mat(R)
    Q_RR = prob_mat(R @ R.T)

    loss_for_space = calc_KL(P_S, Q_RR)

    if t == 0: # t= T の場合
        loss_for_time = lambda1 * calc_KL(P_B, Q_R)
    else:
        X_prev = X_prev.detach()
        R_prev = X_prev @ Y.T
        P_R_prev = prob_mat(R_prev)
        loss_for_time = lambda2 * calc_KL(P_R_prev, Q_R)

    loss = loss_for_space + loss_for_time
    
    return loss

def calc_loss(list_P_S, P_B, list_X, list_Y, lambda1, lambda2):
    loss = 0
    Y = list_Y[0]
    X_prev = None
    for t, (P_S, X) in enumerate(zip(list_P_S, list_X)):
        _loss = calc_loss_at_t(t, P_S, P_B, X, X_prev, Y, lambda1, lambda2)
        loss += _loss
        X_prev = X
        
    return loss


def calc_loss_for_Y(list_P_S, P_B, list_X, list_Y, lambda1, lambda2):
    loss = 0
    Y = list_Y[0]
    X_prev = None
    for t, (P_S, X) in enumerate(zip(list_P_S, list_X)):
        X = X.detach()
        _loss = calc_loss_at_t(t, P_S, P_B, X, X_prev, Y, lambda1, lambda2)
        loss += _loss
        X_prev = X
        
    return loss

def update_list_X(optimizer_list_X, optimizer_list_Y, list_P_S, P_B, list_X, list_Y, lambda1, lambda2):
    X_prev = None
    Y = list_Y[0]
    for t, (P_S, X) in enumerate(zip(list_P_S, list_X)):
        optimizer_list_X.zero_grad() # 勾配をゼロに初期化する
        optimizer_list_Y.zero_grad() # Y は更新しない
        Y = Y.detach()
        loss = calc_loss_at_t(t, P_S, P_B, X, X_prev, Y, lambda1, lambda2)
        optimizer_list_Y.zero_grad() # Y は更新しない
        loss.backward() # KLダイバージェンスの逆伝播を計算する
        optimizer_list_X.step() # パラメータを更新する
        X_prev = list_X[t] # .detach()
        
    # 以下のコードはうまくいかない
    # optimizer_list_X.zero_grad()
    # optimizer_list_Y.zero_grad()
    # loss = calc_loss(list_P_S, P_B, list_X, [Y.detach() for Y in list_Y], lambda1, lambda2)
    # optimizer_list_Y.zero_grad() # Y は更新しない
    # loss.backward() # KLダイバージェンスの逆伝播を計算する
    # optimizer_list_X.step() # パラメータを更新する
        
def update_list_Y(optimizer_list_X, optimizer_list_Y, list_P_S, P_B, list_X, list_Y, lambda1, lambda2):
    optimizer_list_X.zero_grad()
    optimizer_list_Y.zero_grad()
    loss = calc_loss(list_P_S, P_B, [X.detach() for X in list_X], list_Y, lambda1, lambda2)
    optimizer_list_X.zero_grad() # X は更新しない
    loss.backward() # KLダイバージェンスの逆伝播を計算する
    optimizer_list_Y.step() # パラメータを更新する

def whitening(X_org):
    X = (X_org - X_org.mean(dim=0)) / X_org.std(dim=0)
    phi_x = torch.cov(X.t())
    eig = torch.linalg.eigh(phi_x)
    D12 = torch.diag((eig[0] + 1e-12)**(-0.5))
    E = eig[1]
    P = torch.matmul(torch.matmul(E, D12), E.t())
    res = torch.matmul(X, P)
    res = res / torch.sqrt(torch.sum(res**2, dim=1, keepdim=True))
    
    return res

    # Q, R = torch.linalg.qr(X_org)
    # if Q.shape[1] == X_org.shape[1]:
    #     return Q
    # elif R.shape[1] == X_org.shape[1]:
    #     return R
    
    
    
def embed(output_dim, num_objects, B, list_Z, max_epochs, lr, lambda1, lambda2):
    rs = check_random_state(42)
    # device = "cuda:0"
    device = "cpu"
    
    num_references = B.shape[1]

    # 入力
    list_S, C = make_list_S(list_Z)
    list_P_S = prob_mat_list(list_S)
    list_P_S = [torch.from_numpy(P).to(device) for P in list_P_S]
    P_B = prob_mat(B)
    P_B = torch.from_numpy(P_B).to(device)
    
    # 埋め込み
    epoch = 0
    list_X = [] # 対象の座標
    
    # list_X, Y の初期化：ガウス分布で初期化
    T = len(list_Z)
    for t in range(T):
        X_tmp = rs.multivariate_normal(mean=np.zeros(output_dim),cov=np.eye(output_dim), size=num_objects)
        X_tmp = np.array(X_tmp, dtype=floath)
        X_tmp = torch.from_numpy(X_tmp).to(device)
        X_tmp = X_tmp.requires_grad_(True)
        list_X.append(X_tmp)
        
    Y_tmp = rs.multivariate_normal(mean=np.zeros(output_dim),cov=np.eye(output_dim), size=num_references)
    Y_tmp = np.array(Y_tmp, dtype=floath)
    Y_tmp = torch.from_numpy(Y_tmp).to(device)
    Y = Y_tmp.requires_grad_(True)
    list_Y = [Y]

    optimizer_list_X = torch.optim.Adam(list_X, lr=lr)
    optimizer_list_Y = torch.optim.Adam(list_Y, lr=lr)
    
    # 初期の損失を計算
    losses = []
    loss = calc_loss(list_P_S, P_B, list_X, list_Y, lambda1, lambda2)
    losses.append(loss.item())
    print(f"epoch 0: loss={loss}")
    
    # ループ
    for epoch in range(1, max_epochs+1):
        # X の更新: t = T のロス計算 -> 更新, t = T-1 のロス計算 -> 更新, ...
        update_list_X(optimizer_list_X, optimizer_list_Y, list_P_S, P_B, list_X, list_Y, lambda1, lambda2)
        # Y の更新
        update_list_Y(optimizer_list_X, optimizer_list_Y, list_P_S, P_B, list_X, list_Y, lambda1, lambda2)
        with torch.no_grad():
            _Y = whitening(list_Y[0].detach())
        _Y = _Y.requires_grad_(True)
        list_Y = [_Y]
        
        loss = calc_loss(list_P_S, P_B, list_X, list_Y, lambda1, lambda2)
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss={loss}")
    
    return {
        "list_X": [X.detach().cpu().numpy() for X in list_X],
        "Y": [Y.detach().cpu().numpy() for Y in list_Y][0],
        "Ls": losses,
        "C": C
    }
        
    
def log_bilinear(coords_df, label_column, output_dim=2, learning_rate=10, lambda1=0.1, lambda2=0.1, max_epochs=100):

    labels = coords_df[label_column].to_list()

    coords_df = coords_df.drop(label_column)

    B, list_Z, N = make_inputs_for_log_bilinear(coords_df, labels, t_column="t")
    # list_Z: 次元削減前の dataframe t=T,T-1,...,1
    # N: 一時刻あたりの対象の数
    
    result = embed(output_dim, N, B, list_Z, max_epochs, learning_rate, lambda1, lambda2)

    Y = result["Y"]
    list_X = result["list_X"]
    list_X.reverse() 
    losses = result["Ls"]
    
    steps = []
    reduced_X = []

    # label の並び替え
    # column t の 最大値
    # T = coords_df["t"].max()
    # idx_list = coords_df.loc[coords_df[t_column] == T].index.tolist()
    # T_labels = [labels[i] for i in idx_list]
    # sorted_idx = np.argsort(T_labels)

    # # B[T]
    # print("B[T]")
    # B = B[sorted_idx, :]
    # visualize_mat(B, output_dir, "B_T")

    # # R[T]
    # print("R[T]")
    # R = calc_R(list_X[-1], Y)
    # R = R[sorted_idx, :]
    # visualize_mat(R, output_dir, "R_T")

    # # S[T]
    # print("S[T]")
    # C = result["C"]
    # S = calc_S(list_Z[0], C)
    # S = S[sorted_idx, :][:, sorted_idx]
    # visualize_mat(S, output_dir, "S_T")

    # # R2[T]
    # print("R2[T]")
    # R2 = calc_RR(R)
    # R2 = R2[sorted_idx, :][:, sorted_idx]
    # visualize_mat(R2, output_dir, "R2_T")
    
    for epoch, X in enumerate(list_X):
        reduced_X.extend(X.tolist())
        steps.extend([ float(epoch) for i in range(len(X))]) 
        
    steps_df = pl.DataFrame(
        data={'t': steps}
    )
    
    reduced_X_df = pl.DataFrame(
        data=np.array(reduced_X, dtype='float64'),
        schema=[f"dim{i}" for i in range(output_dim)]
    )
    visible_coords_df = pl.concat([steps_df, reduced_X_df], how="horizontal")
    
    return visible_coords_df, Y

class DynamicISNE(BasicDimReducer):
    def __init__(self, output_dim: int = 2, time_oriented_penalty: float = 0.1, last_structure_penalty: float = 0.1):
        self.output_dim = output_dim
        self.time_oriented_penalty = time_oriented_penalty
        self.last_structure_penalty = last_structure_penalty
    
    def fit_transform(self, df: pl.DataFrame, label_column = "label") -> Tuple[pl.DataFrame, np.ndarray]:
        """
        Fit df into an embedded space and return that transformed output.

        Args:
            df (pl.DataFrame): DataFrame to be transformed.
                "t" column is required.
                "dim0", "dim1", ... columns are required.
        """
        coords_df, meta_df = self.split_coords_and_meta(df, extra_cols=[label_column])
        reduced_df, references = log_bilinear(coords_df, label_column, output_dim=self.output_dim, lambda1=self.last_structure_penalty, lambda2=self.time_oriented_penalty)

        return pl.concat([reduced_df, meta_df], how="horizontal"), references