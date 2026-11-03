import copy
import time
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def set_seed(seed):
    """Set random seeds for PyTorch and NumPy."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_regression_model(model, train_loader, val_loader, criterion, optimizer,
                           num_epochs, device, patience=50, scheduler=None,
                           gradual_unfreeze=False, unfreeze_epoch=100,
                           unfreeze_lr_factor=0.5, verbose=False):
    """
    Robust training loop resolving Bugs 1, 4, 5, 6 and logging layer gradient norms (Section 11.3).
    """
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    early_counter = 0

    train_losses = []
    val_losses = []
    val_r2_history = []
    grad_logs = []

    for epoch in range(num_epochs):
        # Gradual unfreezing check for Strategy S3
        if gradual_unfreeze and epoch == unfreeze_epoch:
            if hasattr(model, "unfreeze_all"):
                model.unfreeze_all()
            else:
                for param in model.parameters():
                    param.requires_grad = True

            for param_group in optimizer.param_groups:
                param_group["lr"] *= unfreeze_lr_factor
            if verbose:
                print(f"[Epoch {epoch + 1}] Layers unfrozen and LR adjusted by factor {unfreeze_lr_factor}.")

        # --- Training Step ---
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_train_loss += loss.item() * inputs.size(0)

            loss.backward()

            # Record gradient norms for interpretability (Section 11.3)
            layer_norms = {
                name: float(param.grad.norm().item())
                for name, param in model.named_parameters() if param.grad is not None
            }

            optimizer.step()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        if layer_norms:
            grad_logs.append({"epoch": epoch + 1, **layer_norms})

        # --- Validation Step ---
        model.eval()
        running_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss_val = criterion(outputs, targets)
                running_val_loss += loss_val.item() * inputs.size(0)

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if val_preds:
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)
            r2_per_col = [
                r2_score(val_targets[:, c], val_preds[:, c])
                for c in range(val_targets.shape[1])
            ]
            epoch_val_r2 = float(np.mean(r2_per_col))
        else:
            epoch_val_r2 = float("nan")

        val_r2_history.append(epoch_val_r2)

        if scheduler:
            scheduler.step(epoch_val_loss)

        # Early stopping logic (Bug 1 fix: always deepcopy state_dict)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_counter += 1
            if early_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.6f}")
                break

    # Restore best checkpoint weights (even if loop exhausted num_epochs)
    model.load_state_dict(best_model_state)

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_r2": val_r2_history,
        "best_val_loss": best_val_loss,
        "epochs_run": len(train_losses)
    }

    return model, history, grad_logs
