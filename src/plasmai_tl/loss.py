import torch
import torch.nn as nn


class WeightedMSE:
    """
    Weighted MSE Loss with optional spatial smoothness penalty and L1 regularization.
    """
    def __init__(self, reduction='mean', outputs_points=None, device=None, model=None,
                 l1_lambda=0.0, lambda_smooth=0.0, tau_i=0.0, W_i=None):
        self.reduction = reduction
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if outputs_points is None:
            outputs_points = [i + 1 for i in range(10)]
            
        self.weights = torch.ones(len(outputs_points), dtype=torch.float32).to(self.device)
        self.model = model
        self.l1_lambda = float(l1_lambda)
        self.lambda_smooth = float(lambda_smooth)
        
        num_second_diffs = max(0, len(outputs_points) - 2)
        if isinstance(tau_i, (int, float)):
            self.tau_i = torch.full((num_second_diffs,), float(tau_i), dtype=torch.float32).to(self.device)
        elif tau_i is not None:
            self.tau_i = torch.as_tensor(tau_i, dtype=torch.float32).to(self.device)
        else:
            self.tau_i = torch.zeros(num_second_diffs, dtype=torch.float32).to(self.device)

        if W_i is not None:
            self.W_i = torch.as_tensor(W_i, dtype=torch.float32).to(self.device)
        else:
            self.W_i = torch.ones(num_second_diffs, dtype=torch.float32).to(self.device)

    def __call__(self, pred, target):
        squared_error = ((pred - target) ** 2) * self.weights
        
        if self.reduction == 'mean':
            loss = squared_error.mean()
        elif self.reduction == 'sum':
            loss = squared_error.sum()
        else:
            loss = squared_error

        # Spatially-Aware Smoothness penalty (penalizing jaggedness across output points)
        if self.lambda_smooth > 0 and pred.shape[1] >= 3:
            first_diffs = pred[:, 1:] - pred[:, :-1]
            second_diffs = first_diffs[:, 1:] - first_diffs[:, :-1]
            excess_jaggedness = torch.relu(torch.abs(second_diffs) - self.tau_i)
            weighted_jaggedness = excess_jaggedness * self.W_i
            loss_smooth = torch.mean(weighted_jaggedness ** 2)
            loss = loss + (self.lambda_smooth * loss_smooth)

        if self.model is not None and self.l1_lambda > 0:
            l1_penalty = 0.0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    l1_penalty += torch.norm(param, 1)
            loss = loss + (self.l1_lambda * l1_penalty)

        return loss
