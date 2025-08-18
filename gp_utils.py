from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import torch


def defineSTGP(train_xy, train_z, GP=SingleTaskGP, output_dim=1, state_dict=None, device=torch.device("cpu")):
    gp = GP(train_xy, train_z,
            input_transform=Normalize(d=train_xy.shape[-1]),
            outcome_transform=Standardize(m=output_dim)).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if state_dict is not None:
        gp.load_state_dict(state_dict, strict=False)
    return gp, mll


def defineSaasSTGP(train_X, train_Y, GP=SaasFullyBayesianSingleTaskGP, output_dim=1, state_dict=None, device=torch.device("cpu")):
    gp = GP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=output_dim).to(device)
    )
    if state_dict is not None:
        gp.load_state_dict(state_dict)
    return gp


def perform_inference(gp, X: torch.Tensor, fully_bayesian: bool):
    with torch.no_grad():
        prediction = gp.posterior(X)
    if fully_bayesian:
        return prediction.mixture_mean.squeeze()
    else:
        return prediction.mean
