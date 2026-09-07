from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from data_preprocessing_busbra import get_busbra_dataloaders
from model_busbra import BUSKANModel, FastKANLayer, SplineWeightLayer, count_parameters, seed_everything

#1
@dataclass
class TrainConfig:
    busbra_root: str = "/content/drive/MyDrive/BUS/BUSBRA"
    busbra_csv_filename: str = "5-fold-cv.csv"
    test_fold: int = 1
    save_root: str = "/content/drive/MyDrive/BUS/EKAT_checkpoints_BUSBRA_fold1"

    seed: int = 42
    epochs: int = 150
    batch_size: int = 32
    evaluation_batch_size: int = 8
    num_workers: int = 0

    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 2.0
    mid_learning_rate_factor: float = 2.0
    slow_learning_rate_factor: float = 3.0

    # After four consecutive epochs without a strict increase in image-level
    # test accuracy, multiply every optimizer group's current LR by 0.80.
    lr_scheduler_no_improvement_epochs: int = 4
    lr_scheduler_factor: float = 0.80
    minimum_learning_rate: float = 1.0e-6

    use_amp: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0
    decision_threshold: float = 0.5

    # Save after every N completed batches. Keep 1 for batch-level recovery.
    checkpoint_every_batches: int = 1
    resume: bool = True

    grid_size: int = 64
    spline_order: int = 3

    # Original CIFAR E-KAT regularization terms.
    l1_weight: float = 5.0e-5
    path_competition_weight: float = 4.0e-3
    spline_smoothness_weight: float = 2.0e-2
    attention_l2_weight: float = 1.0e-3
    structure_l2_weight: float = 1.0e-5
    late_structure_l2_weight: float = 7.5e-7
    late_structure_epoch: int = 60
    regularization_warmup_epochs: int = 5


class DeterministicEpochBatchSampler(Sampler[List[int]]):
    """Deterministic shuffled batches with a resumable batch cursor."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0

    @property
    def full_batch_count(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        if not 0 <= start_batch <= self.full_batch_count:
            raise ValueError(f"Invalid batch cursor {start_batch}/{self.full_batch_count}")
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, self.dataset_size, self.batch_size)
        ]
        yield from batches[self.start_batch :]

    def __len__(self) -> int:
        return self.full_batch_count - self.start_batch


def _unpack_batch(batch: Sequence[Any]):
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(f"Expected a 2- or 3-item batch, got {len(batch)} items")


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor. A checkpoint loaded with
    # map_location=device may otherwise move this tensor to CUDA.
    torch.set_rng_state(state["torch"].detach().cpu().to(dtype=torch.uint8))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            item.detach().cpu().to(dtype=torch.uint8)
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _split_fingerprint(train_dataset, test_dataset) -> str:
    """Fingerprint the exact ordered image-level split used by this run."""
    digest = hashlib.sha256()
    for split_name, dataset in (("train", train_dataset), ("test", test_dataset)):
        for record in dataset.records:
            path = Path(record["image_path"])
            stat = path.stat()
            fields = (
                split_name,
                str(record["source_dataset"]),
                str(record["case_id"]),
                str(int(record["label"])),
                str(path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            )
            digest.update(("\x1f".join(fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def calculate_class_weights(train_dataset, device: torch.device) -> torch.Tensor:
    """Balanced CE weights N/(2*N_c), computed only from training images."""
    labels = [int(record["label"]) for record in train_dataset.records]
    counts = Counter(labels)
    if set(counts) != {0, 1}:
        raise ValueError(f"Both classes must occur in training; observed counts={dict(counts)}")
    total = len(labels)
    weights = torch.tensor(
        [total / (2.0 * counts[0]), total / (2.0 * counts[1])],
        dtype=torch.float32,
        device=device,
    )
    return weights


def kan_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    epoch: int,
    class_weights: torch.Tensor,
    config: TrainConfig,
):
    """Class-weighted two-logit CE plus the original E-KAT regularizers."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected model output [B,2], received {tuple(logits.shape)}")
    targets = targets.long()
    ce_loss = F.cross_entropy(logits, targets, weight=class_weights)

    l1_regularization = logits.new_zeros(())
    path_competition = logits.new_zeros(())
    spline_smoothness = logits.new_zeros(())
    attention_l2 = logits.new_zeros(())
    structure_l2 = logits.new_zeros(())

    for module in model.modules():
        if hasattr(module, "spline_weights"):
            weights = module.spline_weights
            l1_regularization = l1_regularization + weights.abs().sum() / module.num_coeffs
            l1_regularization = l1_regularization + module.base_scales.abs().sum()

            second_difference = weights[:, :, 2:] - 2.0 * weights[:, :, 1:-1] + weights[:, :, :-2]
            spline_smoothness = spline_smoothness + second_difference.square().mean()

            edge_strength = weights.abs().mean(dim=-1) + module.base_scales.abs()
            probability = F.softmax(edge_strength / 0.1, dim=0)
            path_competition = path_competition - (
                probability * torch.log(probability + 1.0e-8)
            ).sum() / module.out_neurons

            # Retained for compatibility with the original CIFAR loss. In the
            # current model last_W1 is normally None (or detached for plots).
            if getattr(module, "last_W1", None) is not None:
                attention_l2 = attention_l2 + module.last_W1.square().mean()

        if hasattr(module, "W2") and hasattr(module, "W3_conv"):
            l1_regularization = l1_regularization + module.W2.abs().mean() + module.b.abs().mean()
            l1_regularization = l1_regularization + module.W3_conv.weight.abs().mean()
            structure_l2 = structure_l2 + (
                module.alpha.square()
                + module.beta.square()
                + module.theta.square()
                + module.gamma.square()
            ).sum()

    warmup = min(1.0, (epoch + 1) / max(1, config.regularization_warmup_epochs))
    current_structure_weight = (
        config.late_structure_l2_weight
        if epoch >= config.late_structure_epoch
        else config.structure_l2_weight
    )
    regularization = warmup * (
        config.l1_weight * l1_regularization
        + config.path_competition_weight * path_competition
        + config.spline_smoothness_weight * spline_smoothness
        + config.attention_l2_weight * attention_l2
        + current_structure_weight * structure_l2
    )
    total_loss = ce_loss + regularization
    components = {
        "ce": float(ce_loss.detach()),
        "regularization": float(regularization.detach()),
        "l1": float(l1_regularization.detach()),
        "path_competition": float(path_competition.detach()),
        "spline_smoothness": float(spline_smoothness.detach()),
        "attention_l2": float(attention_l2.detach()),
        "structure_l2": float(structure_l2.detach()),
    }
    return total_loss, components


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    model.eval()
    all_targets: List[int] = []
    all_probabilities: List[float] = []

    for batch in loader:
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = logits.softmax(dim=1)[:, 1]
        all_targets.extend(targets.cpu().numpy().astype(int).tolist())
        all_probabilities.extend(probabilities.cpu().numpy().astype(float).tolist())

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_prob = np.asarray(all_probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "predicted_malignant_fraction": float(y_pred.mean()),
        "confusion_matrix": matrix.tolist(),
        "images": int(y_true.size),
    }
    return metrics


def _structure_summary(model: nn.Module) -> Dict[str, float]:
    values: Dict[str, List[float]] = {
        "alpha": [], "beta": [], "theta": [], "gamma": [], "omiga": [],
        "tau": [], "temperature": [],
    }
    for module in model.modules():
        if isinstance(module, SplineWeightLayer):
            for name in ("alpha", "beta", "theta", "gamma"):
                values[name].append(float(getattr(module, name).detach().abs().mean()))
        if isinstance(module, FastKANLayer):
            values["omiga"].append(float(module.omiga.detach().abs().mean()))
            values["tau"].append(float(module.tau.detach().abs().mean()))
            values["temperature"].append(float(module.temperature.detach().abs().mean()))
    return {name: float(np.mean(items)) if items else float("nan") for name, items in values.items()}


def _save_history(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    flat_rows: List[Dict[str, Any]] = []
    for row in history:
        flat = dict(row)
        flat["confusion_matrix"] = json.dumps(flat["confusion_matrix"])
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _checkpoint_payload(
    *, model, optimizer, scheduler, scaler, config, data_info, class_weights,
    epoch, next_batch_index, accumulator, history, best_test_accuracy,
    best_epoch, early_stop_counter, elapsed_seconds,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "train_accumulator": accumulator,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_test_accuracy": float(best_test_accuracy),
        "best_epoch": int(best_epoch),
        "early_stop_counter": int(early_stop_counter),
        "history": history,
        "elapsed_seconds": float(elapsed_seconds),
        "class_weights": class_weights.detach().cpu(),
        "config": asdict(config),
        "data_info": data_info,
        "rng_state": _capture_rng_state(),
    }


def _verify_resume_config(saved: Dict[str, Any], current: TrainConfig) -> None:
    # Changing any of these invalidates an exact next-batch continuation.
    critical = [
        "busbra_root", "busbra_csv_filename", "test_fold", "seed", "epochs",
        "batch_size", "evaluation_batch_size", "num_workers", "grid_size",
        "spline_order", "learning_rate", "weight_decay",
        "gradient_clip_norm", "use_amp", "decision_threshold",
        "early_stopping_patience", "early_stopping_min_delta",
        "mid_learning_rate_factor", "slow_learning_rate_factor",
        "lr_scheduler_no_improvement_epochs", "lr_scheduler_factor",
        "minimum_learning_rate",
        "l1_weight", "path_competition_weight", "spline_smoothness_weight",
        "attention_l2_weight", "structure_l2_weight",
        "late_structure_l2_weight", "late_structure_epoch",
        "regularization_warmup_epochs",
    ]
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, current={getattr(current, key)!r}"
        for key in critical
        if saved.get(key) != getattr(current, key)
    ]
    if mismatches:
        raise ValueError(
            "Cannot safely resume because critical configuration changed:\n"
            + "\n".join(mismatches)
            + "\nUse a new save_root or set resume=False to start a new experiment."
        )


def train(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    config = config or TrainConfig()
    if config.num_workers != 0:
        raise ValueError("Exact batch-level resume requires num_workers=0")
    if config.test_fold not in {1, 2, 3, 4, 5}:
        raise ValueError("test_fold must be one of {1, 2, 3, 4, 5}")

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    save_dir = Path(config.save_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_path = save_dir / "latest_checkpoint.pt"
    best_path = save_dir / "best_model.pt"
    history_path = save_dir / "training_history.csv"
    results_path = save_dir / "final_results.json"

    # We reuse the preprocessing module's split/transforms, then replace only
    # its training sampler so a batch cursor can be restored without replay.
    original_train_loader, test_loader, data_info = get_busbra_dataloaders(
        busbra_root_dir=config.busbra_root,
        busbra_csv_filename=config.busbra_csv_filename,
        test_fold=config.test_fold,
        batch_size=config.batch_size,
        evaluation_batch_size=config.evaluation_batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        flatten=False,
        return_metadata=True,
    )
    train_dataset = original_train_loader.dataset
    data_info["split_fingerprint"] = _split_fingerprint(
        train_dataset,
        test_loader.dataset,
    )
    batch_sampler = DeterministicEpochBatchSampler(
        len(train_dataset), config.batch_size, config.seed
    )
    loader_generator = torch.Generator().manual_seed(config.seed + 9_000_000)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )

    model = BUSKANModel(
        grid_size=config.grid_size,
        spline_order=config.spline_order,
    ).to(device)
    class_weights = calculate_class_weights(train_dataset, device)
    model.class_weights = class_weights

    slow_keys = ("alpha", "theta", "omiga")
    mid_keys = ("beta", "tau", "temperature", "gamma")
    base_parameters, mid_parameters, slow_parameters = [], [], []
    for name, parameter in model.named_parameters():
        if any(key in name for key in slow_keys):
            slow_parameters.append(parameter)
        elif any(key in name for key in mid_keys):
            mid_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    optimizer = AdamW(
        [
            {"params": base_parameters, "lr": config.learning_rate, "name": "base"},
            {
                "params": mid_parameters,
                "lr": config.learning_rate * config.mid_learning_rate_factor,
                "name": "mid_structure",
            },
            {
                "params": slow_parameters,
                "lr": config.learning_rate * config.slow_learning_rate_factor,
                "name": "slow_structure",
            },
        ],
        weight_decay=config.weight_decay,
    )
    if config.lr_scheduler_no_improvement_epochs < 1:
        raise ValueError("lr_scheduler_no_improvement_epochs must be at least 1")
    if not 0.0 < config.lr_scheduler_factor < 1.0:
        raise ValueError("lr_scheduler_factor must be strictly between 0 and 1")
    if config.minimum_learning_rate < 0.0:
        raise ValueError("minimum_learning_rate must be non-negative")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler_factor,
        # ReduceLROnPlateau reduces when num_bad_epochs > patience. Therefore,
        # patience=3 performs the reduction after exactly four bad epochs.
        patience=config.lr_scheduler_no_improvement_epochs - 1,
        threshold=0.0,
        threshold_mode="abs",
        min_lr=config.minimum_learning_rate,
    )
    scaler = _make_grad_scaler(amp_enabled)

    start_epoch = 0
    start_batch = 0
    accumulator: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []
    best_test_accuracy = -float("inf")
    best_epoch = 0
    early_stop_counter = 0
    prior_elapsed_seconds = 0.0

    if config.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        _verify_resume_config(checkpoint["config"], config)
        saved_fingerprint = checkpoint.get("data_info", {}).get("split_fingerprint")
        current_fingerprint = data_info["split_fingerprint"]
        if saved_fingerprint != current_fingerprint:
            raise ValueError(
                "Cannot safely resume because the BUS-BRA image files or "
                "the official fold split changed. Use a new "
                "save_root or set resume=False to start a new experiment."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["next_batch_index"])
        accumulator = checkpoint.get("train_accumulator")
        history = checkpoint.get("history", [])
        best_test_accuracy = float(checkpoint.get("best_test_accuracy", -float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        prior_elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        _restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"Resumed exactly at epoch {start_epoch + 1}, "
            f"next batch {start_batch + 1}/{batch_sampler.full_batch_count}."
        )

    model.eval()
    with torch.inference_mode():
        smoke_output = model(torch.zeros(1, 4, 96, 96, device=device))
    initial_metrics = evaluate(model, test_loader, device, config.decision_threshold)
    total_parameters, trainable_parameters = count_parameters(model)

    print("=" * 100)
    print("BUS-BRA binary E-KAT training")
    print(f"Device: {device} | AMP: {amp_enabled}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input shape: (B,4,96,96) | smoke output: {tuple(smoke_output.shape)}")
    print(
        f"Parameters: total={total_parameters:,} | "
        f"trainable={trainable_parameters:,}"
    )
    print(
        f"Images: total={data_info['total_images']} | "
        f"train={data_info['train_images']} (benign={data_info['train_benign']}, "
        f"malignant={data_info['train_malignant']}) | "
        f"test={data_info['test_images']} (benign={data_info['test_benign']}, "
        f"malignant={data_info['test_malignant']})"
    )
    print(
        f"Official fold split: train folds={data_info['training_folds']} | "
        f"test fold={data_info['official_test_fold']}"
    )
    print(f"Class weights [benign, malignant]: {class_weights.detach().cpu().tolist()}")
    print(
        "Learning rates: "
        + ", ".join(f"{group['name']}={group['lr']:.3e}" for group in optimizer.param_groups)
    )
    print(f"Checkpoint directory: {save_dir}")
    print(f"Resume checkpoint exists: {latest_path.exists()}")
    print(
        f"[Initial test] Acc={initial_metrics['accuracy']*100:.2f}% "
        f"BalAcc={initial_metrics['balanced_accuracy']*100:.2f}% "
        f"ROC-AUC={initial_metrics['roc_auc']:.4f} PR-AUC={initial_metrics['pr_auc']:.4f}"
    )
    print("=" * 100)

    run_started = time.perf_counter()
    stopped_early = early_stop_counter >= config.early_stopping_patience
    if stopped_early:
        print(
            "The restored run had already reached the early-stopping limit; "
            "no additional epoch will be trained."
        )
        start_epoch = config.epochs
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        resume_batch = start_batch if epoch == start_epoch else 0
        batch_sampler.set_epoch(epoch, resume_batch)

        if resume_batch > 0:
            if accumulator is None:
                raise RuntimeError("Checkpoint lacks the partial-epoch accumulator")
            train_state = dict(accumulator)
        else:
            train_state = {
                "loss_sum": 0.0,
                "ce_sum": 0.0,
                "regularization_sum": 0.0,
                "correct": 0.0,
                "images": 0.0,
            }

        model.train()
        remaining = len(train_loader)
        progress = tqdm(
            train_loader,
            total=remaining,
            initial=0,
            desc=f"Epoch {epoch+1}/{config.epochs} (from batch {resume_batch+1})",
        )
        for local_index, batch in enumerate(progress):
            absolute_batch_index = resume_batch + local_index
            images, targets, _ = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                logits = model(images)
                loss, loss_parts = kan_binary_loss(
                    logits, targets, model, epoch, class_weights, config
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_images = targets.numel()
            train_state["loss_sum"] += float(loss.detach()) * batch_images
            train_state["ce_sum"] += loss_parts["ce"] * batch_images
            train_state["regularization_sum"] += loss_parts["regularization"] * batch_images
            train_state["correct"] += float((logits.argmax(dim=1) == targets).sum())
            train_state["images"] += float(batch_images)
            next_batch = absolute_batch_index + 1

            progress.set_postfix(
                loss=f"{train_state['loss_sum']/train_state['images']:.4f}",
                ce=f"{train_state['ce_sum']/train_state['images']:.4f}",
                acc=f"{100*train_state['correct']/train_state['images']:.2f}%",
            )

            if (
                next_batch % config.checkpoint_every_batches == 0
                or next_batch == batch_sampler.full_batch_count
            ):
                elapsed = prior_elapsed_seconds + time.perf_counter() - run_started
                payload = _checkpoint_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config=config, data_info=data_info, class_weights=class_weights,
                    epoch=epoch, next_batch_index=next_batch,
                    accumulator=train_state, history=history,
                    best_test_accuracy=best_test_accuracy, best_epoch=best_epoch,
                    early_stop_counter=early_stop_counter, elapsed_seconds=elapsed,
                )
                _atomic_torch_save(payload, latest_path)

        if train_state["images"] <= 0:
            raise RuntimeError("No training images were processed")

        test_metrics = evaluate(model, test_loader, device, config.decision_threshold)
        train_loss = train_state["loss_sum"] / train_state["images"]
        train_ce = train_state["ce_sum"] / train_state["images"]
        train_reg = train_state["regularization_sum"] / train_state["images"]
        train_accuracy = train_state["correct"] / train_state["images"]
        structure = _structure_summary(model)
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        epoch_seconds = time.perf_counter() - epoch_started

        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_regularization": train_reg,
            "train_accuracy": train_accuracy,
            **test_metrics,
            **{f"structure_{key}": value for key, value in structure.items()},
            "lr_base": current_lrs[0],
            "lr_mid": current_lrs[1],
            "lr_slow": current_lrs[2],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        _save_history(history, history_path)

        previous_best = best_test_accuracy
        save_as_best = test_metrics["accuracy"] >= previous_best
        early_stopping_improvement = (
            test_metrics["accuracy"] >= previous_best + config.early_stopping_min_delta
        )
        if save_as_best:
            best_test_accuracy = test_metrics["accuracy"]
            best_epoch = epoch + 1
            best_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "test_metrics": test_metrics,
                "best_test_accuracy": best_test_accuracy,
                "config": asdict(config),
                "data_info": data_info,
                "class_weights": class_weights.detach().cpu(),
            }
            _atomic_torch_save(best_payload, best_path)

        # Equality is treated as an early-stopping improvement as requested.
        if early_stopping_improvement:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print("-" * 100)
        print(
            f"[Epoch {epoch+1}] train loss={train_loss:.5f} | weighted CE={train_ce:.5f} | "
            f"regularization={train_reg:.5f} | train Acc={train_accuracy*100:.2f}%"
        )
        print(
            f"[Image-level test] Acc={test_metrics['accuracy']*100:.2f}% "
            f"BalAcc={test_metrics['balanced_accuracy']*100:.2f}% "
            f"Sensitivity/Recall={test_metrics['sensitivity_recall']*100:.2f}% "
            f"Specificity={test_metrics['specificity']*100:.2f}% "
            f"Precision={test_metrics['precision']*100:.2f}% "
            f"F1={test_metrics['f1']*100:.2f}% "
            f"ROC-AUC={test_metrics['roc_auc']:.4f} PR-AUC={test_metrics['pr_auc']:.4f}"
        )
        print(f"[Image-level test] confusion matrix [[TN,FP],[FN,TP]]: {test_metrics['confusion_matrix']}")
        print(
            "Structure means: "
            + " ".join(f"{key}={value:.4f}" for key, value in structure.items())
        )
        connectivity = model.get_active_conn_info()
        print(
            "Connectivity: "
            + " | ".join(
                f"{name}={info['active']}/{info['total']} ({info['ratio']*100:.1f}%), "
                f"tau={info['tau_mean']:.4f}"
                for name, info in connectivity.items()
            )
        )
        print(
            f"LR(base/mid/slow)={current_lrs[0]:.3e}/{current_lrs[1]:.3e}/{current_lrs[2]:.3e} | "
            f"epoch time={epoch_seconds/60:.2f} min | best test Acc={best_test_accuracy*100:.2f}% "
            f"at epoch {best_epoch} | patience={early_stop_counter}/{config.early_stopping_patience}"
        )
        if save_as_best:
            relation = ">" if test_metrics["accuracy"] > previous_best else "="
            print(f"Best model updated because current test Acc {relation} historical best: {best_path}")

        scheduler.step(test_metrics["accuracy"])
        elapsed = prior_elapsed_seconds + time.perf_counter() - run_started
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config=config, data_info=data_info, class_weights=class_weights,
            epoch=epoch + 1, next_batch_index=0, accumulator=None, history=history,
            best_test_accuracy=best_test_accuracy, best_epoch=best_epoch,
            early_stop_counter=early_stop_counter, elapsed_seconds=elapsed,
        )
        _atomic_torch_save(payload, latest_path)

        start_batch = 0
        accumulator = None
        if early_stop_counter >= config.early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping: test accuracy stayed below the historical best for "
                f"{config.early_stopping_patience} completed epochs."
            )
            break

    total_elapsed = prior_elapsed_seconds + time.perf_counter() - run_started
    if not best_path.exists():
        raise RuntimeError("Training ended without producing best_model.pt")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_metrics = evaluate(model, test_loader, device, config.decision_threshold)
    results = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_test_metrics": final_metrics,
        "stopped_early": stopped_early,
        "completed_epochs": len(history),
        "training_seconds": total_elapsed,
        "best_model_path": str(best_path),
        "latest_checkpoint_path": str(latest_path),
        "history_path": str(history_path),
        "config": asdict(config),
        "data_info": data_info,
    }
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, ensure_ascii=False, indent=2)
    print("=" * 100)
    print(f"Training finished | best epoch={results['best_epoch']} | test Acc={final_metrics['accuracy']*100:.2f}%")
    print(f"Best model: {best_path}")
    print(f"Latest resumable checkpoint: {latest_path}")
    print(f"History: {history_path} | Results: {results_path}")
    print(f"Total accumulated training time: {total_elapsed/3600:.2f} h")
    return results


if __name__ == "__main__":
    train(TrainConfig())
