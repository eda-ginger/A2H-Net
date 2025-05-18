import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression
from math import sqrt

def calculate_regression_metrics(pred: torch.Tensor, real: torch.Tensor) -> dict:
    """
    Calculates various regression metrics.

    Args:
        pred (torch.Tensor): Predicted values from the model. Expected to be 1D or squeezable to 1D.
        real (torch.Tensor): Actual (ground truth) values. Expected to be 1D or squeezable to 1D.

    Returns:
        dict: A dictionary containing the calculated metrics:
              'mse' (Mean Squared Error),
              'rmse' (Root Mean Squared Error),
              'mae' (Mean Absolute Error),
              'r2' (R-squared),
              'pcc' (Pearson Correlation Coefficient),
              'ci' (Concordance Index),
              'sd' (Standard Deviation of residuals).
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(real, torch.Tensor):
        raise TypeError("Inputs 'pred' and 'real' must be PyTorch Tensors.")

    # Detach from graph, flatten, and ensure they are on the same device
    pred_detached = pred.detach().flatten()
    real_detached = real.detach().flatten()

    if pred_detached.shape != real_detached.shape:
        raise ValueError(
            f"Shape mismatch after flattening: pred shape {pred_detached.shape}, real shape {real_detached.shape}"
        )
    
    num_samples = pred_detached.numel()

    if num_samples == 0: # Handle empty tensors
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'pcc': np.nan,
            'ci': np.nan,
            'sd': np.nan
        }

    # Convert to NumPy for calculations
    pred_np = pred_detached.cpu().numpy()
    real_np = real_detached.cpu().numpy()

    # MSE (Mean Squared Error)
    mse = F.mse_loss(pred_detached, real_detached).item()

    # RMSE (Root Mean Squared Error) - using the provided implementation
    rmse = sqrt(((real_np - pred_np)**2).mean(axis=0))

    # MAE (Mean Absolute Error) - using the provided implementation
    mae = (np.abs(real_np - pred_np)).mean()

    # R-squared
    r2 = np.nan
    if num_samples >= 2:
        try:
            r2 = r2_score(real_np, pred_np)
        except ValueError:
            pass

    # Pearson Correlation Coefficient - using the provided implementation
    pcc = np.nan
    if num_samples >= 2:
        try:
            pcc = np.corrcoef(real_np, pred_np)[0,1]
            if np.isnan(pcc):
                pcc = np.nan
        except ValueError:
            pass
            
    # Concordance Index
    ci = np.nan
    if num_samples >= 2:
        try:
            if not (np.isnan(real_np).any() or np.isnan(pred_np).any()):
                if len(np.unique(real_np)) > 1 and len(np.unique(pred_np)) > 1:
                    ci_val = concordance_index(real_np, pred_np)
                    if not np.isnan(ci_val):
                        ci = ci_val
        except Exception:
            pass       
    
    # Standard Deviation of residuals - using the provided implementation
    sd_val = np.nan
    if num_samples >= 2:
        try:
            f, y = pred_np.reshape(-1,1), real_np.reshape(-1,1)
            lr = LinearRegression()
            lr.fit(f, y)
            y_ = lr.predict(f)
            sd_val = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
        except Exception:
            pass
            
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pcc': pcc,
        'ci': ci,
        'sd': sd_val
    }


if __name__ == '__main__':
    # Example Usage
    predictions = torch.tensor([1.0, 2.5, 3.0, 4.5, 5.0])
    actuals = torch.tensor([1.2, 2.3, 3.5, 4.0, 5.3])
    
    metrics = calculate_regression_metrics(predictions, actuals)
    # General Case: Expect high R2, PCC, CI if predictions are good.
    print("Calculated Metrics (General Case):")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n--- Edge Case: Single Sample ---")
    predictions_single = torch.tensor([1.0])
    actuals_single = torch.tensor([1.2])
    metrics_single = calculate_regression_metrics(predictions_single, actuals_single)
    # Single Sample: R2, PCC, CI are undefined (NaN) as they require at least 2 samples.
    # MSE, RMSE, MAE can be calculated based on the single difference.
    print("Calculated Metrics (Single Sample):")
    for key, value in metrics_single.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: Constant Actuals ---")
    predictions_const_actual = torch.tensor([2.0, 2.1, 1.9, 2.05])
    actuals_const = torch.tensor([2.0, 2.0, 2.0, 2.0])
    metrics_const_actual = calculate_regression_metrics(predictions_const_actual, actuals_const)
    # Constant Actuals: 
    # R2 is typically 0.0 or negative if predictions vary (model explains no variance, or performs worse than mean).
    # PCC is undefined (NaN) as actuals have no variance.
    # CI is undefined (NaN) as all actuals are tied, no unique order.
    print("Calculated Metrics (Constant Actuals):")
    for key, value in metrics_const_actual.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: Constant Predictions ---")
    predictions_const_pred = torch.tensor([2.0, 2.0, 2.0, 2.0])
    actuals_for_const_pred = torch.tensor([1.0, 2.0, 3.0, 0.5])
    metrics_const_pred = calculate_regression_metrics(predictions_const_pred, actuals_for_const_pred)
    # Constant Predictions:
    # R2 is typically 0.0 or negative if actuals vary (model explains no variance, or performs worse than mean).
    # PCC is undefined (NaN) as predictions have no variance.
    # CI is undefined (NaN) as all predictions are tied, no unique order for them.
    print("Calculated Metrics (Constant Predictions):")
    for key, value in metrics_const_pred.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: Perfect Correlation ---")
    predictions_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0])
    actuals_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0])
    metrics_perfect = calculate_regression_metrics(predictions_perfect, actuals_perfect)
    # Perfect Correlation: All error metrics (MSE, RMSE, MAE) are 0.
    # R2 = 1.0 (all variance explained).
    # PCC = 1.0 (perfect positive linear correlation).
    # CI = 1.0 (perfect order agreement).
    print("Calculated Metrics (Perfect Correlation):")
    for key, value in metrics_perfect.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: Perfect Negative Correlation ---")
    predictions_neg_perfect = torch.tensor([4.0, 3.0, 2.0, 1.0]) # pred: 4,3,2,1
    actuals_neg_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0])   # real: 1,2,3,4
    metrics_neg_perfect = calculate_regression_metrics(predictions_neg_perfect, actuals_neg_perfect)
    # Perfect Negative Correlation:
    # R2 can be highly negative (e.g., -3.0 for these values) if predictions are systematically 
    # on the opposite side of the mean of actuals, indicating a very poor fit despite perfect negative linear correlation.
    # PCC = -1.0 (perfect negative linear correlation).
    # CI = 0.0 (perfect inverse order agreement).
    print("Calculated Metrics (Perfect Negative Correlation):")
    for key, value in metrics_neg_perfect.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: No Correlation (Random) ---")
    predictions_random = torch.tensor([1.5, 3.2, 0.5, 4.8, 2.0])
    actuals_random = torch.tensor([2.0, 1.0, 4.0, 3.0, 5.0])
    metrics_random = calculate_regression_metrics(predictions_random, actuals_random)
    # No Correlation (Random):
    # R2 is close to 0.0 or negative (model does not explain variance, or performs worse than mean).
    # PCC is close to 0.0 (no strong linear correlation, can vary with small random samples).
    # CI is close to 0.5 (random order agreement).
    print("Calculated Metrics (No Correlation - Random):")
    for key, value in metrics_random.items():
        print(f"{key}: {value}")
        
    print("\n--- Edge Case: Non-linear Relationship (U-shape example) ---")
    actuals_nonlinear = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) # Actuals: U-shape (or inverted V)
    predictions_nonlinear = torch.tensor([1.0, 0.5, 0.1, 0.5, 1.0]) # Predictions: Inverted U-shape (or V)
    metrics_nonlinear = calculate_regression_metrics(actuals_nonlinear, predictions_nonlinear)
    # Non-linear Relationship (U-shape vs Inverted U-shape):
    # R2 can be very low (negative), as a linear fit is extremely poor for opposing U-shapes.
    # PCC can be strongly negative if the overall trend of one U-shape opposes the other (e.g., one generally decreases then increases, the other increases then decreases across the range).
    # CI would be low (e.g., 0.0 for this specific data) because the rank orders are largely opposite.
    print("Calculated Metrics (Non-linear Relationship - U-shape):")
    for key, value in metrics_nonlinear.items():
        print(f"{key}: {value}")
        
    print("\n--- Edge Case: Few Data Points with Outlier ---")
    actuals_outlier = torch.tensor([1.0, 2.0, 3.0, 15.0]) # 15.0 is an outlier
    predictions_outlier = torch.tensor([1.2, 2.3, 2.8, 5.0]) # Prediction for outlier is not as extreme
    metrics_outlier = calculate_regression_metrics(actuals_outlier, predictions_outlier)
    # Few Data Points with Outlier:
    # MSE, RMSE, MAE will be heavily influenced by the outlier if not predicted well.
    # R2 can decrease significantly (become negative) due to the outlier if it harms the linear fit.
    # PCC can be skewed by the outlier but may remain high if the outlier and other points still suggest a linear trend.
    # CI can remain high (e.g., 1.0 for this data) if the outlier does not change the rank order of predictions relative to actuals.
    print("Calculated Metrics (Few Data Points with Outlier):")
    for key, value in metrics_outlier.items():
        print(f"{key}: {value}")

    print("\n--- Edge Case: Empty Input ---")
    predictions_empty = torch.tensor([])
    actuals_empty = torch.tensor([])
    metrics_empty = calculate_regression_metrics(predictions_empty, actuals_empty)
    # Empty Input: All metrics are undefined (NaN).
    print("Calculated Metrics (Empty Input):")
    for key, value in metrics_empty.items():
        print(f"{key}: {value}") 