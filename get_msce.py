import os
import random
import argparse
import numpy as np
import torch

from lcpfn.bar_distribution import get_bucket_limits, FullSupportBarDistribution
from lcpfn.transformer import TransformerModel
# from lcpfn.transformer_noupward import TransformerModel
from data.get_test_data import *
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Optional, Union

def bardist_calibration(
    bar_dist,              # BarDistribution instance
    logits,    # Shape: (N, num_bars) - Model output logits
    y_true,    # Shape: (N,) - Ground truth values
    confidence_levels: Optional[List[float]] = None
) -> Dict:
    """
    Compute Mean squareolute Calibration Error (mce) for BarDistribution.
    
    Parameters:
    -----------
    bar_dist : BarDistribution
        Instance of the BarDistribution class
    logits : np.ndarray or torch.Tensor
        Model output logits, shape (N, num_bars)
    y_true : np.ndarray or torch.Tensor
        Ground truth values, shape (N,)
    confidence_levels : list of float, optional
        Confidence levels to evaluate (default: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        
    Returns:
    --------
    metrics : dict
        Dictionary containing calibration metrics with mean_square_miscalibration (mce) as key metric
    """
    # Default confidence levels if not provided
    if confidence_levels is None:
        confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Convert numpy arrays to torch tensors if needed
    import torch
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits, dtype=torch.float32)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    
    # Ensure correct shapes
    if len(y_true.shape) == 2:  # (N, 1)
        y_true = y_true.squeeze(dim=1)  # -> (N,)
    
    # Number of data points
    n_data = y_true.shape[0]
    
    # Initialize results
    observed_coverages = []
    interval_widths = []
    
    # For each confidence level, compute interval and check coverage
    for conf_level in confidence_levels:
        alpha = 1.0 - conf_level
        lower_quantile = alpha / 2
        upper_quantile = 1.0 - lower_quantile
        
        # Compute prediction intervals using icdf (inverse CDF) method
        lower_bounds = bar_dist.icdf(logits, lower_quantile)
        upper_bounds = bar_dist.icdf(logits, upper_quantile)
        
        # Check if ground truth is within interval
        within_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage = torch.mean(within_interval.float()).item()
        
        # Calculate average interval width
        width = torch.mean(upper_bounds - lower_bounds).item()
        
        # Store results
        observed_coverages.append(coverage)
        interval_widths.append(width)

        # print(lower_bounds)
        # print("Lower bounds:", lower_bounds[:5])  # First 5 values
        # print("Upper bounds:", upper_bounds[:5])
        # print("Y true:", y_true[:5])
        # print("Within interval:", within_interval[:5])
        # print("Coverage:", coverage)
    
    # Calculate miscalibration metrics
    square_miscalibrations = [np.square(observed - expected) 
                          for observed, expected in zip(observed_coverages, confidence_levels)]
    
    # Compile metrics
    metrics = {
        'confidence_levels': confidence_levels,
        'observed_coverages': observed_coverages,
        'interval_widths': interval_widths,
        'square_miscalibrations': square_miscalibrations,
        'mean_square_miscalibration': np.mean(square_miscalibrations),  # This is the mce
        'max_square_miscalibration': max(square_miscalibrations)
    }
    
    return metrics


def predict_quantiles(logits, qs, criterion):
    return torch.stack([criterion.icdf(logits.squeeze(), q) for q in qs], dim=1)

def get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True, epsilon=1e-2):
    xc, yc, xt, yt = xc.unsqueeze(0).to(device), yc.unsqueeze(0).to(device), xt.unsqueeze(0).to(device), yt.unsqueeze(0).to(device)
    
    if y_normalize: # Normalize y, since DD and nano is not normalized unlike bench
        y_max = max(yc.max().item(), yt.max().item())
        yc, yt = yc / y_max, yt / y_max

    # apply min-max normalization to [epsilon, 1]
    x_min, x_max = min(xc.min().item(), xt.min().item()), max(xc.max().item(), xt.max().item())
    xc = (xc - x_min) / (x_max - x_min) * (1-epsilon) + epsilon
    xt = (xt - x_min) / (x_max - x_min) * (1-epsilon) + epsilon

    # forward
    yt_pred = model(xc, yc, xt)
    mce = bardist_calibration(criterion, yt_pred.squeeze(0), yt.squeeze(0))['mean_square_miscalibration']
    negative_log_prob = criterion(yt_pred, yt)
    loss = negative_log_prob.sum().item()
    yt_pred = criterion.median(yt_pred)

    # rmsle and log-likelihood
    rmsle = torch.sqrt(((torch.log(yt_pred) - torch.log(yt)) ** 2).sum() / yt.numel()).item()
    log_likelihood = -(negative_log_prob.mean().item())
            
    return loss, rmsle, log_likelihood, mce

def get_pred(device, model, criterion, xc, yc, xt, yt, pred_all=False, y_normalize=True, epsilon=1e-2):
    xc, yc, xt, yt = xc.unsqueeze(0).to(device), yc.unsqueeze(0).to(device), xt.unsqueeze(0).to(device), yt.unsqueeze(0).to(device)
    
    if y_normalize: # Normalize y, since DD and nano is not normalized unlike bench
        y_max = yc.max().item()
        yc, yt = yc / y_max, yt / y_max

    # apply min-max normalization
    x_min, x_max = min(xc.min().item(), xt.min().item()), max(xc.max().item(), xt.max().item())
    xc = (xc - x_min) / (x_max - x_min) * (1-epsilon) + epsilon
    xt = (xt - x_min) / (x_max - x_min) * (1-epsilon) + epsilon
    
    x, y = torch.cat([xc, xt], dim=1), torch.cat([yc, yt], dim=1)

    if pred_all:
        xt = x
        yt = y

    yt_pred = model(xc, yc, xt)
    if yt_pred.shape[1] > 1:
        predictions = predict_quantiles(yt_pred, qs=[0.05, 0.5, 0.95], criterion=criterion)
        predictions = predictions.cpu().detach().numpy()
    else:
        predictions = None

    yt_pred = criterion.median(yt_pred)

    if y_normalize:
        yt_pred = yt_pred * y_max
        predictions = predictions * y_max if predictions is not None else None

    # convert to numpy
    yt_pred = yt_pred.cpu().detach().squeeze().numpy()

    return yt_pred, predictions

def plot_and_log_bench(logger, device, model, criterion, data, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        str_key = '_'.join(key).replace('/','-')

        loss, rmsle, log_likelihood, mce = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=False)
        yt_pred, predictions = get_pred(device, model, criterion, xc, yc, xt, yt, pred_all=pred_all, y_normalize=False)

        log.append((*key, loss, rmsle, log_likelihood, mce))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    avg_mce = np.mean([l[6] for l in log])
    std_mce = np.std([l[6] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}', f'{avg_mce:.4f}+-{std_mce:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood', 'mce'])

    # # save log
    # df.to_csv(os.path.join(save_dir, f"{domain}.csv" if cutoff == -1 else f"{domain}_{cutoff}.csv"), index=False)

    # print log
    # print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    # print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    # print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print(f"{domain} AVG mce: {avg_mce:.4f} +/- {std_mce:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)
        logger.meter(domain, 'mce', avg_mce)

    return df

def plot_and_log_DD(logger, device, model, criterion, data, labels, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        task = key[1]

        loss, rmsle, log_likelihood, mce = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True)

        log.append((*key, loss, rmsle, log_likelihood, mce))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    avg_mce = np.mean([l[6] for l in log])
    std_mce = np.std([l[6] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}', f'{avg_mce:.4f}+-{std_mce:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood', 'mce'])

    # # save log
    # df.to_csv(os.path.join(save_dir, f"{domain}.csv" if cutoff == -1 else f"{domain}_{cutoff}.csv"), index=False)

    # print log
    # print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    # print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    # print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print(f"{domain} AVG mce: {avg_mce:.4f} +/- {std_mce:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)
        logger.meter(domain, 'mce', avg_mce)

    return df

def plot_and_nano_bench(logger, device, model, criterion, data, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        task = key[1]

        loss, rmsle, log_likelihood, mce = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True)
        log.append((*key, loss, rmsle, log_likelihood, mce))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    avg_mce = np.mean([l[6] for l in log])
    std_mce = np.std([l[6] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}', f'{avg_mce:.4f}+-{std_mce:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood', 'mce'])

    # # save log
    # df.to_csv(os.path.join(save_dir, f"{domain}.csv" if cutoff == -1 else f"{domain}_{cutoff}.csv"), index=False)

    # print log
    # print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    # print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    # print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print(f"{domain} AVG mce: {avg_mce:.4f} +/- {std_mce:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)
        logger.meter(domain, 'mce', avg_mce)

    return df

def plot_and_log_colpret(logger, device, model, criterion, data, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        model_name = key[2]

        loss, rmsle, log_likelihood, mce = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True)

        log.append((*key, loss, rmsle, log_likelihood, mce))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    avg_mce = np.mean([l[6] for l in log])
    std_mce = np.std([l[6] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}', f'{avg_mce:.4f}+-{std_mce:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood', 'mce'])

    # print log
    # print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    # print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    # print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print(f"{domain} AVG mce: {avg_mce:.4f} +/- {std_mce:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)
        logger.meter(domain, 'mce', avg_mce)

    return df

def test(logger, device, model, criterion, cutoff, data_dir, save_dir):
    # make directory
    os.makedirs(save_dir, exist_ok=True)

    # test data
    IC_data, NMT_data, LM_data, BB_data = get_bench_data(data_dir, cutoff=cutoff)
    DD_data, DD_labels = get_DD_data(data_dir, cutoff=cutoff)
    nano_data = get_nano_data(data_dir, cutoff=cutoff)
    colpret_data = get_colpret_data(data_dir, cutoff=cutoff)

    IC_df = plot_and_log_bench(logger, device, model, criterion, IC_data, save_dir, cutoff=cutoff, pred_all=False)
    NMT_df = plot_and_log_bench(logger, device, model, criterion, NMT_data, save_dir, cutoff=cutoff, pred_all=False)
    LM_df = plot_and_log_bench(logger, device, model, criterion, LM_data, save_dir, cutoff=cutoff, pred_all=False)
    BB_df = plot_and_log_bench(logger, device, model, criterion, BB_data, save_dir, cutoff=cutoff, pred_all=False)
    DD_df = plot_and_log_DD(logger, device, model, criterion, DD_data, DD_labels, save_dir, cutoff=cutoff, pred_all=False)
    nano_df = plot_and_nano_bench(logger, device, model, criterion, nano_data, save_dir, cutoff=cutoff, pred_all=False)
    colpret_df = plot_and_log_colpret(logger, device, model, criterion, colpret_data, save_dir, cutoff=cutoff, pred_all=False)

    # merge df and save
    dataframes = [IC_df, BB_df, LM_df, NMT_df, DD_df, nano_df, colpret_df]

    # Separate main parts and last rows
    main_parts = [df.iloc[:-1] for df in dataframes]  # Exclude the last row of each dataframe
    last_rows = [df.iloc[-1:] for df in dataframes]   # Only the last row of each dataframe

    # Concatenate main parts and last rows
    merged_df = pd.concat(main_parts, ignore_index=True)    # Merge main parts first
    merged_df = pd.concat([merged_df] + last_rows, ignore_index=True)  # Add last rows at the end

    # Save to a file
    merged_df.to_csv(os.path.join(save_dir, "CE.csv" if cutoff == -1 else f"CE_cutoff{cutoff}.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=42)

    # dir
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--checkpoint_dir', type=str, default="/workspace/scaling-law-2025/pretrained_surrogate_results/def1030/seed1/")
    parser.add_argument('--exp_name', type=str, default="val01")

    # hparams for data
    parser.add_argument('--cutoff', type=float, default=-1)

    # hparams for model
    parser.add_argument('--d_output', type=int, default=1000)
    parser.add_argument('--nlayers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    cutoff = args.cutoff
    logger = None

    os.environ["WANDB_SILENT"] = "true"
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.device(device)
    save_dir = os.path.join(args.checkpoint_dir, 'vis')

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    

    # model and opt
    model = TransformerModel(
        d_output=args.d_output,
        d_model=args.d_model,
        dim_feedforward=2*args.d_model,
        nlayers=args.nlayers,
        dropout=args.dropout,
        activation="gelu",
        y_stats=(torch.tensor(0.5), torch.tensor(0.5)),
    ).to(device)

    borders = get_bucket_limits(num_outputs=args.d_output, full_range=(0., 1.), ys=None)
    criterion = FullSupportBarDistribution(borders).to(device)

    model_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'model.pt'), map_location=device)
    model.load_state_dict(model_checkpoint)

    criterion_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'criterion.pt'), map_location=device)
    criterion.load_state_dict(criterion_checkpoint)
    model.eval()


    test(logger, device, model, criterion, cutoff, data_dir, save_dir)