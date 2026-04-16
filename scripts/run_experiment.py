import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DatasetBuilder
from src.data.sentence_builder import SentenceBuilder
from src.evaluation.evaluator import run_evaluation
from src.models.rnn_model import SequenceModel
from src.training.trainer import Trainer
from src.utils.device import get_device

VALID_VARIANTS = [
    "baseline_rnn",
    "variant_a_rnn",
    "variant_b_rnn",
    "baseline_lstm",
    "variant_a_lstm",
    "variant_b_lstm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment variant.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=VALID_VARIANTS,
        help="Experiment variant ID",
    )
    return parser.parse_args()


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Inject project-root-relative output paths into a flat "paths" key
    cfg["paths"] = {
        "output_dir": str(PROJECT_ROOT / cfg["output"].get("checkpoints_dir", "output").split("/")[0]),
        "checkpoints_dir": str(PROJECT_ROOT / cfg["output"]["checkpoints_dir"]),
        "logs_dir": str(PROJECT_ROOT / cfg["output"]["logs_dir"]),
    }
    # Resolve output_dir to just "output/"
    cfg["paths"]["output_dir"] = str(PROJECT_ROOT / "output")
    return cfg


def main() -> None:
    args = parse_args()
    variant_id = args.variant

    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    variant_cfg = config["variants"][variant_id]

    architecture = variant_cfg["architecture"]
    predict_n = variant_cfg["predict_n"]
    model_seed = training_cfg["model_seed"]
    data_seed = data_cfg["data_seed"]

    device = get_device()
    print(f"[device] {device}")

    # Step 5: Build sentences
    sentence_builder = SentenceBuilder(data_cfg, str(PROJECT_ROOT))
    sentences, final_seed = sentence_builder.build()
    print(f"[data] {len(sentences):,} sentences built (final_seed={final_seed})")

    # Step 6: Build datasets (DataLoader construction happens here — before model seed)
    dataset_builder = DatasetBuilder(
        sentences=sentences,
        final_seed=final_seed,
        predict_n=predict_n,
        data_cfg=data_cfg,
        batch_size=training_cfg["batch_size"],
    )
    train_loader, eval_loader, test_loader, vocab = dataset_builder.build()
    vocab_size = len(vocab)
    print(f"[data] vocab_size={vocab_size}  train={len(train_loader.dataset):,}  "
          f"eval={len(eval_loader.dataset):,}  test={len(test_loader.dataset):,}")

    # Step 7: Set model seed AFTER DataLoader construction
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    random.seed(model_seed)

    # Step 8: Instantiate model
    model = SequenceModel(
        vocab_size=vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        architecture=architecture,
        predict_n=predict_n,
    ).to(device)
    print(f"[model] {architecture.upper()} predict_n={predict_n} "
          f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Step 9: Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        training_cfg=training_cfg,
        variant_id=variant_id,
        device=device,
        logs_dir=config["paths"]["logs_dir"],
        checkpoints_dir=config["paths"]["checkpoints_dir"],
    )
    trainer.run()

    # Step 10: Evaluate
    trainer_stats = {
        "best_eval_loss": trainer.best_eval_loss,
        "best_eval_loss_epoch": trainer.best_eval_loss_epoch,
        "total_epochs": training_cfg["epochs"],
        "model_seed": model_seed,
        "data_seed": data_seed,
    }
    metrics = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        variant_id=variant_id,
        config=config,
        trainer_stats=trainer_stats,
    )
    print("\n[results]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
