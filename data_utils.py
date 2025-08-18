from sklearn.model_selection import train_test_split
from torch.quasirandom import SobolEngine
import torch
import pandas as pd
import os
import numpy as np


def split_data(X: pd.DataFrame, Y: pd.DataFrame, n_initial: int = 3, seed: int = 42, device="cpu"):
    init_size = n_initial/len(X)
    X_train = torch.DoubleTensor(X.values)
    Y_train = torch.DoubleTensor(Y.values)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train, Y_train, test_size=1-init_size, random_state=seed)

    return X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)


def load_training_data(file_path: str, x_cols: list[str], y_cols: list[str]):
    data = pd.read_csv(file_path, delimiter=",")
    X = torch.DoubleTensor(data[x_cols].values)
    Y = torch.DoubleTensor(data[y_cols].values)
    return X, Y


def save_gp_model(model, parent_path: str, file_name: str) -> None:
    '''Save model as .pth file
    '''
    if not (os.path.isdir(parent_path)):
        os.mkdir(parent_path)
    if isinstance(model, dict):
        torch.save(model, os.path.join(parent_path, file_name))
    else:
        torch.save(model.state_dict(), os.path.join(parent_path, file_name))


def save_score(score: np.array, parent_path: str, file_name: str) -> None:
    if not (os.path.isdir(parent_path)):
        os.mkdir(parent_path)
    if torch.is_tensor(score[0]):
        score = np.array([tensor.detach().numpy() for tensor in score])
    np.save(os.path.join(parent_path, file_name), score)


def sobol_single_point(dim: int, min: int = 0, max: int = 1, constraints=None):
    if type(min) != torch.Tensor:
        min = torch.Tensor([min])
    if type(max) != torch.Tensor:
        max = torch.Tensor([max])
    sobol = SobolEngine(dimension=dim, scramble=True)
    point = sobol.draw(1, dtype=torch.float64)
    point = min + (max - min) * point
    if constraints == None:
        return point
    for con in constraints:
        if check_if_point_within_bounds(con, point):
            pass
        else:
            point = sobol_single_point(dim, min, max, constraints)
            break
    return point


def sobol_sample(num_points: int, dim: int, min: int = 0, max: int = 1, constraints=None):
    X_sample = torch.empty((num_points, dim), dtype=torch.int64)
    for i in range(num_points):
        X_sample[i] = sobol_single_point(
            dim=dim, min=min, max=max, constraints=constraints)
    return X_sample
