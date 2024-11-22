import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("hyperparameter_tuning_results.csv")

    # describe the data
    print(data.describe())

    # Plot each hyperparameter against the r2 val and test score
    hyperparameters = ["num_layers", "neurons", "batch_size", "epochs"]
    accuracy_metrics = [
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_r2",
        "val_r2",
        "test_r2",
    ]

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, hyperparameter in enumerate(hyperparameters):
        sns.lineplot(
            data=data, x=hyperparameter, y="val_r2", ax=axes[i], label="Validation"
        )
        sns.lineplot(
            data=data, x=hyperparameter, y="train_r2", ax=axes[i], label="Train"
        )
        sns.scatterplot(
            data=data,
            x=hyperparameter,
            y="val_r2",
            ax=axes[i],
            alpha=0.1,
            label=None,
            marker="o",
            s=5,
        )
        sns.scatterplot(
            data=data,
            x=hyperparameter,
            y="test_r2",
            ax=axes[i],
            alpha=0.1,
            label=None,
            marker="o",
            s=5,
        )
        axes[i].set_title(f"Impact of {hyperparameter} on R2 score")
        axes[i].set_xlabel(hyperparameter)
        axes[i].set_ylabel("R2 score")
        axes[i].set_ylim(0.6, 0.85)
        axes[i].legend(loc="lower right")

    plt.tight_layout(pad=3.0)
    plt.show()
