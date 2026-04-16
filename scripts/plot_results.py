"""Generate training curve plots and results table screenshot for iter2."""
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


def plot_curves(variants: list[str], title: str, out_path: Path) -> None:
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


def plot_results_table(out_path: Path) -> None:
    results_csv = PROJECT_ROOT / "output" / "analysis" / "results_summary.csv"
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

    plt.title("Iteration 2 — Structured Sentences Results", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cross_iteration_comparison(out_path: Path) -> None:
    iter1_csv = PROJECT_ROOT / "output" / "analysis" / "results_summary_iter1_random.csv"
    iter2_csv = PROJECT_ROOT / "output" / "analysis" / "results_summary.csv"

    df1 = pd.read_csv(iter1_csv)
    df2 = pd.read_csv(iter2_csv)

    variants = df1["variant_id"].tolist()
    x = range(len(variants))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Iteration 1 (Random) vs Iteration 2 (Structured) — All 6 Variants",
                 fontsize=13, fontweight="bold")

    bars1 = ax1.bar([i - width/2 for i in x], df1["test_loss"], width, label="Iter 1 (random)", color="#e74c3c", alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], df2["test_loss"], width, label="Iter 2 (structured)", color="#2ecc71", alpha=0.8)
    ax1.axhline(y=9.21, color="red", linestyle="--", alpha=0.5, label="ln(10k)=9.21 (iter1 floor)")
    ax1.axhline(y=8.59, color="green", linestyle="--", alpha=0.5, label="ln(5404)=8.59 (iter2 floor)")
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Variant")
    ax1.set_ylabel("Test Loss")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(["p=1\nRNN", "p=2\nRNN", "p=3\nRNN", "p=1\nLSTM", "p=2\nLSTM", "p=3\nLSTM"], fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar([i - width/2 for i in x], df1["test_seq_acc"], width, label="Iter 1 (random)", color="#e74c3c", alpha=0.8)
    ax2.bar([i + width/2 for i in x], df2["test_seq_acc"], width, label="Iter 2 (structured)", color="#2ecc71", alpha=0.8)
    ax2.set_title("Test Sequence Accuracy")
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Seq Acc")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(["p=1\nRNN", "p=2\nRNN", "p=3\nRNN", "p=1\nLSTM", "p=2\nLSTM", "p=3\nLSTM"], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_curves(RNN_VARIANTS, "Iteration 2 — RNN Variants (Structured Sentences)",
                SCREENSHOTS_DIR / "iter2_training_curves_rnn.png")
    plot_curves(LSTM_VARIANTS, "Iteration 2 — LSTM Variants (Structured Sentences)",
                SCREENSHOTS_DIR / "iter2_training_curves_lstm.png")
    plot_results_table(SCREENSHOTS_DIR / "iter2_results_table.png")
    plot_cross_iteration_comparison(SCREENSHOTS_DIR / "iter2_results_comparison.png")
    print("All plots saved.")
