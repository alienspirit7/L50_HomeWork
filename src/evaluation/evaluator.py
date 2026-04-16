import csv
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


def run_evaluation(
    model,
    test_loader,
    device,
    variant_id: str,
    config: dict,
    trainer_stats: dict,
) -> dict:
    """Evaluate model on test set and append results to results_summary.csv."""

    variant_cfg = config["variants"][variant_id]
    architecture = variant_cfg["architecture"]
    predict_n = variant_cfg["predict_n"]

    model.eval()
    total_loss = 0.0
    total_seq_correct = 0
    total_seq_count = 0

    with torch.no_grad():
        for input_ids, targets in test_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_seq_correct += (preds == targets).all(dim=-1).float().sum().item()
            total_seq_count += targets.size(0)

    test_loss = total_loss / max(len(test_loader), 1)
    test_seq_acc = total_seq_correct / max(total_seq_count, 1)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_timestamp = datetime.utcnow().isoformat()

    metrics = {
        "variant_id": variant_id,
        "architecture": architecture,
        "predict_n": predict_n,
        "test_loss": test_loss,
        "test_seq_acc": test_seq_acc,
        "best_eval_loss": trainer_stats["best_eval_loss"],
        "best_eval_loss_epoch": trainer_stats["best_eval_loss_epoch"],
        "total_epochs": trainer_stats["total_epochs"],
        "model_params": model_params,
        "data_seed": trainer_stats["data_seed"],
        "model_seed": trainer_stats["model_seed"],
        "run_timestamp": run_timestamp,
    }

    columns = [
        "variant_id", "architecture", "predict_n", "test_loss", "test_seq_acc",
        "best_eval_loss", "best_eval_loss_epoch", "total_epochs",
        "model_params", "data_seed", "model_seed", "run_timestamp",
    ]

    results_path = Path(config["paths"]["output_dir"]) / "analysis" / "results_summary.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(columns)
        writer.writerow([metrics[c] for c in columns])

    print(
        f"[eval] {variant_id} test_loss={test_loss:.4f} "
        f"test_seq_acc={test_seq_acc:.4f} params={model_params:,}"
    )
    return metrics
