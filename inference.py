import os
import random
import argparse
import numpy as np
import torch
import pandas as pd
from pfn.bar_distribution import get_bucket_limits, FullSupportBarDistribution
from pfn.transformer import TransformerModel
from data.get_test_data import get_bench_data, get_DD_data, get_nano_data
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def predict_quantiles(logits, qs, criterion):
    return torch.stack([criterion.icdf(logits.squeeze(), q) for q in qs], dim=1)

def get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True, epsilon=1e-2):
    xc, yc, xt, yt = xc.unsqueeze(0).to(device), yc.unsqueeze(0).to(device), xt.unsqueeze(0).to(device), yt.unsqueeze(0).to(device)
    
    if y_normalize: # Normalize y, since DD and nano is not normalized unlike bench
        y_max = yc.max().item()
        yc, yt = yc / y_max, yt / y_max

    # apply min-max normalization to [epsilon, 1]
    x_min, x_max = min(xc.min().item(), xt.min().item()), max(xc.max().item(), xt.max().item())
    xc = (xc - x_min) / (x_max - x_min) * (1-epsilon) + epsilon
    xt = (xt - x_min) / (x_max - x_min) * (1-epsilon) + epsilon

    # forward
    yt_pred = model(xc, yc, xt)
    negative_log_prob = criterion(yt_pred, yt)
    loss = negative_log_prob.sum().item()
    yt_pred = criterion.median(yt_pred)

    # rmsle and log-likelihood
    rmsle = torch.sqrt(((torch.log(yt_pred) - torch.log(yt)) ** 2).sum() / yt.numel()).item()
    log_likelihood = -(negative_log_prob.mean().item())
            
    return loss, rmsle, log_likelihood

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
        plt.figure()
        str_key = '_'.join(key).replace('/','-')

        loss, rmsle, log_likelihood = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=False)
        yt_pred, predictions = get_pred(device, model, criterion, xc, yc, xt, yt, pred_all=pred_all, y_normalize=False)

        # plot data
        plt.plot(xc, yc, color='black', label="context", marker='o')
        plt.scatter(xt, yt, color=[0.0, 0.925, 0.0], label="target", marker='o')
        
        # plot PFN
        if predictions is not None:
            plt.plot(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, "blue", label="NSL-PFN")
            plt.fill_between(
                    xt if not pred_all else torch.cat([xc, xt], dim=0), predictions[:, 0], predictions[:, 2], color="blue", alpha=0.2, label="CI of 90%"
            )
        else:
            plt.scatter(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, color="blue", label="NSL-PFN")
        
        # plot cutoff
        plt.vlines(max(xc), 0, 1, linewidth=0.5, color="k", label="cutoff", linestyle='--')

        plt.xscale("log")
        plt.ylim(0, 1.1)
        plt.xlabel("Training Data Size")
        plt.ylabel("Test Error Rate")

        plt.title(f"{key}\nrmsle:{rmsle:.4f} / log-likelihood:{log_likelihood:.4f}")
        plt.legend()
        figure_save_path = os.path.join(save_dir, domain, f"{str_key}.png" if cutoff == -1 else f"{str_key}_{cutoff}.png")
        plt.savefig(figure_save_path)
        plt.close()

        log.append((*key, loss, rmsle, log_likelihood))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood'])

    # print log
    print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)

    return df

def plot_and_log_DD(logger, device, model, criterion, data, labels, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        plt.figure()
        task = key[1]

        loss, rmsle, log_likelihood = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True)
        yt_pred, predictions = get_pred(device, model, criterion, xc, yc, xt, yt, pred_all=pred_all, y_normalize=True)

        # plot data
        plt.plot(xc, yc, color='black', label="context", marker='o')
        plt.scatter(xt, yt, color=[0.0, 0.925, 0.0], label="target", marker='o')
        
        # plot PFN
        if predictions is not None:
            plt.plot(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, "blue", label="NSL-PFN")
            plt.fill_between(
                    xt if not pred_all else torch.cat([xc, xt], dim=0), predictions[:, 0], predictions[:, 2], color="blue", alpha=0.2, label="CI of 90%"
            )
        else:
            plt.scatter(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, color="blue", label="NSL-PFN")
        
        # plot cutoff
        plt.vlines(max(xc), 0, 1, linewidth=0.5, color="k", label="cutoff", linestyle='--')

        x_min, x_max, y_min, y_max = min(xc.min().item(), xt.min().item()), max(xc.max().item(), xt.max().item()), min(yc.min().item(), yt.min().item()), max(yc.max().item(), yt.max().item())
        plt.xlim(x_min*.865,x_max*1.05)
        plt.ylim(y_min*.9, y_max*1.05)
        x_label, y_label = labels[task]
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.title(f"{key}\nrmsle:{rmsle:.4f} / log-likelihood:{log_likelihood:.4f}")
        plt.legend()
        figure_save_path = os.path.join(save_dir, domain, f"{task}.png" if cutoff == -1 else f"{task}_{cutoff}.png")
        plt.savefig(figure_save_path)
        plt.close()

        log.append((*key, loss, rmsle, log_likelihood))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood'])

    # print log
    print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)

    return df

def plot_and_nano_bench(logger, device, model, criterion, data, save_dir, cutoff=-1, pred_all=False):
    domain = data[0][0][0]
    os.makedirs(os.path.join(save_dir, domain), exist_ok=True)
    log = []
    for i, (key, xc, yc, xt, yt) in enumerate(data):
        plt.figure()
        task = key[1]

        loss, rmsle, log_likelihood = get_metric(device, model, criterion, xc, yc, xt, yt, y_normalize=True)
        yt_pred, predictions = get_pred(device, model, criterion, xc, yc, xt, yt, pred_all=pred_all, y_normalize=True)

        # plot data
        plt.plot(xc, yc, color='black', label="context", marker='o')
        plt.scatter(xt, yt, color=[0.0, 0.925, 0.0], label="target", marker='o')
        
        # plot PFN
        if predictions is not None:
            plt.plot(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, "blue", label="NSL-PFN")
            plt.fill_between(
                    xt if not pred_all else torch.cat([xc, xt], dim=0), predictions[:, 0], predictions[:, 2], color="blue", alpha=0.2, label="CI of 90%"
            )
        else:
            plt.scatter(xt if not pred_all else torch.cat([xc, xt], dim=0), yt_pred, color="blue", label="NSL-PFN")
        
        # plot cutoff
        plt.vlines(max(xc), 0, 1, linewidth=0.5, color="k", label="cutoff", linestyle='--')

        x_min, x_max, y_min, y_max = min(xc.min().item(), xt.min().item()), max(xc.max().item(), xt.max().item()), min(yc.min().item(), yt.min().item()), max(yc.max().item(), yt.max().item())
        plt.xscale("log")
        plt.xlim(x_min*.865,x_max*1.05)
        plt.ylim(y_min*.9, y_max*1.05)
        plt.xlabel('n_embed')
        plt.ylabel('val_loss')

        plt.title(f"{key}\nrmsle:{rmsle:.4f} / log-likelihood:{log_likelihood:.4f}")
        plt.legend()
        figure_save_path = os.path.join(save_dir, domain, f"{task}.png" if cutoff == -1 else f"{task}_{cutoff}.png")
        plt.savefig(figure_save_path)
        plt.close()

        log.append((*key, loss, rmsle, log_likelihood))

    # get average and std of rmsle and log-likelihood
    avg_loss = np.mean([l[3] for l in log])
    std_loss = np.std([l[3] for l in log])
    avg_rmsle = np.mean([l[4] for l in log])
    std_rmsle = np.std([l[4] for l in log])
    avg_log_likelihood = np.mean([l[5] for l in log])
    std_log_likelihood = np.std([l[5] for l in log])
    log.append((domain, '', 'AVG', f'{avg_loss:.4f}+-{std_loss:.4f}', f'{avg_rmsle:.4f}+-{std_rmsle:.4f}', f'{avg_log_likelihood:.4f}+-{std_log_likelihood:.4f}'))
    df = pd.DataFrame(log, columns=['domain', 'task', 'model', 'loss', 'rmsle', 'log-likelihood'])

    # print log
    print(f"{domain} AVG Loss: {avg_loss:.4f} +/- {std_loss:.4f}")
    print(f"{domain} AVG RMSLE: {avg_rmsle:.4f} +/- {std_rmsle:.4f}")
    print(f"{domain} AVG Log-likelihood: {avg_log_likelihood:.4f} +/- {std_log_likelihood:.4f}")
    print()

    # log to wandb
    if logger is not None:
        logger.meter(domain, 'loss', avg_loss)
        logger.meter(domain, 'rmsle', avg_rmsle)
        logger.meter(domain, 'log-likelihood', avg_log_likelihood)

    return df

def test(logger, device, model, criterion, cutoff, data_dir, save_dir):
    # make directory
    os.makedirs(save_dir, exist_ok=True)

    # test data
    IC_data, NMT_data, LM_data, BB_data = get_bench_data(data_dir, cutoff=cutoff)
    DD_data, DD_labels = get_DD_data(data_dir, cutoff=cutoff)
    nano_data = get_nano_data(data_dir, cutoff=cutoff)

    IC_df = plot_and_log_bench(logger, device, model, criterion, IC_data, save_dir, cutoff=cutoff, pred_all=True)
    NMT_df = plot_and_log_bench(logger, device, model, criterion, NMT_data, save_dir, cutoff=cutoff, pred_all=True)
    LM_df = plot_and_log_bench(logger, device, model, criterion, LM_data, save_dir, cutoff=cutoff, pred_all=True)
    BB_df = plot_and_log_bench(logger, device, model, criterion, BB_data, save_dir, cutoff=cutoff, pred_all=True)
    DD_df = plot_and_log_DD(logger, device, model, criterion, DD_data, DD_labels, save_dir, cutoff=cutoff, pred_all=True)
    nano_df = plot_and_nano_bench(logger, device, model, criterion, nano_data, save_dir, cutoff=cutoff, pred_all=True)

    # merge df and save
    dataframes = [IC_df, BB_df, LM_df, NMT_df, DD_df, nano_df]

    # Separate main parts and last rows
    main_parts = [df.iloc[:-1] for df in dataframes]  # Exclude the last row of each dataframe
    last_rows = [df.iloc[-1:] for df in dataframes]   # Only the last row of each dataframe

    # Concatenate main parts and last rows
    merged_df = pd.concat(main_parts, ignore_index=True)    # Merge main parts first
    merged_df = pd.concat([merged_df] + last_rows, ignore_index=True)  # Add last rows at the end

    # Save to a file
    merged_df.to_csv(os.path.join(save_dir, "log.csv" if cutoff == -1 else f"log_cutoff{cutoff}.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=42)

    # dir
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--checkpoint_dir', type=str, default="./pretrained_surrogate_results/default")

    # hparams for data
    parser.add_argument('--cutoff', type=float, default=-1)

    # hparams for model
    parser.add_argument('--d_output', type=int, default=1000)
    parser.add_argument('--nlayers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    cutoff = args.cutoff
    logger = None

    os.environ["WANDB_SILENT"] = "true"
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.device(device)
    save_dir = os.path.join(args.checkpoint_dir, f'cutoff{str(cutoff)}')

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