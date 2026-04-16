import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class Trainer:
    """Trains a SequenceModel and logs metrics to CSV + checkpoints."""

    def __init__(
        self,
        model,
        train_loader,
        eval_loader,
        training_cfg: dict,
        variant_id: str,
        device,
        logs_dir: str,
        checkpoints_dir: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.training_cfg = training_cfg
        self.variant_id = variant_id
        self.device = device
        self.logs_dir = Path(logs_dir)
        self.checkpoints_dir = Path(checkpoints_dir)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=training_cfg["learning_rate"]
        )
        self.epochs = training_cfg["epochs"]
        self.grad_clip = training_cfg.get("grad_clip", 0.0)
        self.save_every_n_epochs = training_cfg.get("save_every_n_epochs", 1)

        self.best_eval_loss = float("inf")
        self.best_eval_loss_epoch = 0

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ckpt_variant_dir = self.checkpoints_dir / variant_id
        ckpt_variant_dir.mkdir(parents=True, exist_ok=True)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        return F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=0,
        )

    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_seq_correct = 0
        total_seq_count = 0

        with torch.no_grad():
            for input_ids, targets in self.eval_loader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(input_ids)
                loss = self._compute_loss(logits, targets)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                total_seq_correct += (preds == targets).all(dim=-1).float().sum().item()
                total_seq_count += targets.size(0)

        avg_loss = total_loss / max(len(self.eval_loader), 1)
        seq_acc = total_seq_correct / max(total_seq_count, 1)
        return avg_loss, seq_acc

    def run(self) -> None:
        csv_path = self.logs_dir / f"{self.variant_id}.csv"
        write_header = not csv_path.exists()
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(
                ["epoch", "train_loss", "eval_loss", "eval_seq_acc",
                 "learning_rate", "elapsed_seconds"]
            )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            self.model.train()
            total_train_loss = 0.0

            for input_ids, targets in self.train_loader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(input_ids)
                loss = self._compute_loss(logits, targets)
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_train_loss += loss.item()

            train_loss = total_train_loss / max(len(self.train_loader), 1)
            eval_loss, eval_seq_acc = self._eval_epoch()
            elapsed = time.time() - epoch_start

            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_eval_loss_epoch = epoch

            lr = self.optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, eval_loss, eval_seq_acc, lr, elapsed])
            csv_file.flush()

            save = (epoch % self.save_every_n_epochs == 0) or (epoch == self.epochs)
            if save:
                ckpt_path = self.checkpoints_dir / self.variant_id / f"epoch_{epoch:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "eval_seq_acc": eval_seq_acc,
                    },
                    ckpt_path,
                )

            print(
                f"[{self.variant_id}] epoch {epoch}/{self.epochs} "
                f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} "
                f"eval_seq_acc={eval_seq_acc:.4f} ({elapsed:.1f}s)"
            )

        csv_file.close()
