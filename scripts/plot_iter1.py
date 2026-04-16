"""Generate iter1 training curve plots and results table from current output/logs/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "output" / "logs"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

RNN_VARIANTS = ["baseline_rnn", "variant_a_rnn", "variant_b_rnn"]
LSTM_VARIANTS = ["baseline_lstm", "variant_a_lstm", "variant_b_lstm"]
LABELS = {
    "baseline_rnn": "RNN predict_n=1",
    "variant_a_rnn": "RNN predict_n=2",
    "variant_b_rnn": "RNN predict_n=3",
    "baseline_lstm": "LSTM predict_n=1",
    "variant_a_lstm": "LSTM predict_n=2",
    "variant_b_lstm": "LSTM predict_n=3",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_curves(variants: list, title: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, variant_id in enumerate(variants):
        csv = LOGS_DIR / f"{variant_id}.csv"
        df = pd.read_csv(csv)
        epochs = df["epoch"]
        color = COLORS[i]
        label = LABELS[variant_id]
        ax1.plot(epochs, df["train_loss"], color=color, linestyle="--", alpha=0.5)
        ax1.plot(epochs, df["eval_loss"], color=color, linestyle="-", label=label)
        ax2.plot(epochs, df["eval_seq_acc"], color=color, linestyle="-", label=label)

    ax1.set_title("Train loss (dashed) / Eval loss (solid)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Eval sequence accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Seq Acc")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_results_table(results_csv: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(results_csv)
    display = df[["variant_id", "architecture", "predict_n",
                  "test_loss", "test_seq_acc", "best_eval_loss", "best_eval_loss_epoch"]].copy()
    display.columns = ["Variant", "Arch", "predict_n",
                       "Test Loss", "Seq Acc", "Best Eval Loss", "Best Eval Epoch"]
    display["Test Loss"] = display["Test Loss"].round(4)
    display["Seq Acc"] = display["Seq Acc"].apply(lambda x: f"{x:.4f}")
    display["Best Eval Loss"] = display["Best Eval Loss"].round(4)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    header_color = "#2c3e50"
    rnn_color = "#d6eaf8"
    lstm_color = "#d5f5e3"
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row <= 3:
            cell.set_facecolor(rnn_color)
        else:
            cell.set_facecolor(lstm_color)

    plt.title(title, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_curves(RNN_VARIANTS, "Iteration 1 — RNN Variants (Random Sentences)",
                SCREENSHOTS_DIR / "iter1_training_curves_rnn.png")
    plot_curves(LSTM_VARIANTS, "Iteration 1 — LSTM Variants (Random Sentences)",
                SCREENSHOTS_DIR / "iter1_training_curves_lstm.png")
    plot_results_table(
        PROJECT_ROOT / "output" / "analysis" / "results_summary.csv",
        SCREENSHOTS_DIR / "iter1_results_table.png",
        "Iteration 1 — Random Sentences Results",
    )
    print("All iter1 plots saved.")
