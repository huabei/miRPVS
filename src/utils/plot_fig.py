import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def plot_fig(preds, targets):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, preds, s=1)
    ax.set_xlabel("experimental")
    ax.set_ylabel("predicted")
    # ax.set_xlim(-10, -4)
    # ax.set_ylim(-10, -4)
    ax.plot([-15, 0], [-15, 0], color="black", linestyle="--")
    return fig
