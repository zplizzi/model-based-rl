from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_distribution(data):
    N = len(data)
    assert data.shape == (N, )
    y = np.arange(0, N) / N
    assert y.shape == (N, )
    # fig = go.Figure(data=go.Scatter(x=data, y=y, mode='markers'))
    # plt.plot(data, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data, y)
    return fig
    # fig.show()



def quantile_regression_loss(delta, tau):
    rho = (tau - (delta < 0).int()).abs()

    # Normal version (non-huber)
    # return rho * delta.abs()

    # Huber-loss version
    k = .1
    sq = .5 * delta**2
    lin = k * (delta.abs() - .5 * k)
    idxs = delta.abs() < k
    lin[idxs] = sq[idxs]
    # huber = torch.nn.functional.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
    return rho * lin


def distributional_loss(pred, target):
    # Inputs should be the N distribution supports
    batch_size = len(pred)
    N = pred.shape[-1]
    assert pred.shape == (batch_size, N)
    assert target.shape == (batch_size, N)

    # The midpoints of the desired quantiles
    t_hat = torch.arange(.5, N + .5, 1.0).cuda() / N

    # I want pairs of t_hat[i], (target[j] - pred[i])
    # target should get broadcast along the last dimension
    deltas = target.unsqueeze(1) - pred.unsqueeze(2).expand(
        batch_size, N, N)
    # So basically in deltas we have for each item in batch:
    # a row will have pred_i and a column will have target_j
    assert deltas.shape == (batch_size, N, N)
    # So we expand t_hat such that it has the same value across each row
    # so that each t_hat[i] is associated with pred[i]

    # The raw output has shape (batch_size, N, N). Mean over dim=1 gives
    # the prediction loss for that quantile. Sum over the new dim=1 gives
    # the total loss for all quantiles.
    # So we're left with shape (batch_size, )
    loss = quantile_regression_loss(
        deltas,
        t_hat.unsqueeze(1).expand(N, N)).mean(dim=1).sum(dim=1)
    assert loss.shape == (batch_size, )

    return loss
