# %%
import os

import matplotlib.pyplot as plt
import numpy as np


def load_loss_history(log_file, num_batches):
    epoch_end = f"{num_batches-1}/{num_batches}"
    with open(log_file) as f:
        logs = f.readlines()
    loss_batch, loss_epoch, lrs = [], [], []
    for line in logs:
        if line.startswith("Epoch: "):
            loss_batch.append(float(line[:-1].split("\t")[-1].split()[1]))
            if epoch_end in line:
                loss_epoch.append(
                    float(line[:-1].split("\t")[-1].split()[2][1:-1])
                )
        if line.startswith("lr "):
            lrs.append(float(line.split()[-1]))
    return loss_batch, loss_epoch, lrs


def plot_loss(loss_batch, loss_epoch, filename=None):
    steps = np.arange(0, len(loss_batch))
    fig = plt.figure()
    ax = plt.gca()
    ax_step = ax.twiny()
    ax_step.plot(steps, loss_batch, c="C0", alpha=0.2, label="", zorder=0)
    ax_step.xaxis.set_major_formatter("{x:,.0f}")
    if ax_step is not ax:
        ax_step.grid(False)
    ax.plot(loss_epoch, c="C0", lw=2.0, label="Train loss", zorder=1)
    lowest_idx = np.array(loss_epoch).argmin()
    ax.plot(
        lowest_idx,
        loss_epoch[lowest_idx],
        "*r",
        label=f"Lowest epoch={lowest_idx}",
    )
    n_epochs = len(loss_epoch)
    ax.set_xlim([-(n_epochs + 1) * 0.05, n_epochs * 1.05])
    ax.set_xlabel("# of Epoch")
    ax_step.set_xlabel("# of Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    if filename is not None:
        fig.savefig(filename)
        plt.close()


def plot_lr(lrs, num_batches, filename=None):
    fig = plt.figure()
    ax = plt.gca()
    ax_step = ax.twiny()
    ax_step.plot(lrs)
    ax_step.xaxis.set_major_formatter("{x:,.0f}")
    if ax_step is not ax:
        ax_step.grid(False)
    n_epochs = len(lrs) / num_batches
    ax.set_xlim([-(n_epochs + 1) * 0.05, n_epochs * 1.05])
    ax.set_xlabel("# of Epoch")
    ax_step.set_xlabel("# of Iteration")
    ax.set_ylabel("Learning rate")
    if filename is not None:
        fig.savefig(filename)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--num-batches", type=int, default=420)
    args = parser.parse_args()
    log_file = os.path.join(args.output_dir, "log.txt")
    loss_batch, loss_epoch, lrs = load_loss_history(log_file, args.num_batches)
    plot_loss(
        loss_batch, loss_epoch, os.path.join(args.output_dir, "loss.png")
    )
    plot_lr(lrs, args.num_batches, os.path.join(args.output_dir, "lr.png"))
