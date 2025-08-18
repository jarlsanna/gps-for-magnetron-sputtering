import torch
from sklearn.metrics import root_mean_squared_error


def calculate_rmse(gp, X, y_true, fully_bayesian=False):
    with torch.no_grad():
        posterior = gp.posterior(X)
        if fully_bayesian:
            posterior_mean = posterior.mixture_mean.squeeze()
        else:
            posterior_mean = posterior.mean
    if X.is_cuda:
        rmse = torch.sqrt(torch.mean((posterior_mean - y_true) ** 2))
        return rmse.item()
    else:
        rmse = root_mean_squared_error(y_true.detach().numpy(),
                      posterior_mean.detach().numpy())
        return rmse
