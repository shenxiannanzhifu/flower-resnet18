"""v0.4/v0.5 训练入口：训练、验证、保存最佳模型、重载并预测参考图片。

项目根目录运行：.venv/Scripts/python.exe scripts/train.py --device cuda
默认仍是 200/100 张、3 轮 smoke。正式实验必须显式指定 --config configs/train_formal.json，
并选择新的 --run-dir / --report-dir。formal 使用完整 train/val，始终不读测试图像。
不支持断点续训；报告和模型目录非空时拒绝覆盖。
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint import check_reference_prediction, load_checkpoint, save_checkpoint
from src.data import FlowerDataset, read_json, seed_everything
from src.engine import run_epoch
from src.model import ResNet18


def write_json(path: Path, value: dict) -> None:
    """保存实际运行配置或结果；ensure_ascii=False 保留中文说明。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_config(config: dict) -> None:
    """区分正式/试运行，拒绝将带子集抽样的配置标为完整正式实验。"""
    if config["purpose"] not in ("smoke", "formal"):
        raise ValueError("purpose 必须为 smoke 或 formal。")
    if config["purpose"] == "formal" and any(name in config for name in ("train_per_class", "val_per_class")):
        raise ValueError("formal 必须使用完整 train/val，不能设置 per_class 抽样。")
    names = ["epochs", "batch_size", "scheduler_step_size"]
    if config["purpose"] == "smoke":
        names += ["train_per_class", "val_per_class"]
    for name in names:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} 必须为正整数。")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("seed 必须是 0 到 2**32-1 的整数。")
    for name in ("learning_rate", "momentum", "weight_decay", "scheduler_gamma"):
        if not math.isfinite(config[name]):
            raise ValueError(f"{name} 必须为有限数值。")
    if (config["learning_rate"] <= 0 or not 0 <= config["momentum"] < 1
            or config["weight_decay"] < 0 or not 0 < config["scheduler_gamma"] <= 1):
        raise ValueError("学习率、动量、权重衰减或学习率调度系数超出合法范围。")


def select_smoke_indices(dataset: FlowerDataset, per_class: int, seed: int) -> list:
    """在既有子集内部每类固定抽样；不跨 split、不按模型效果挑样本。"""
    indices = []
    for name, label in sorted(dataset.class_to_idx.items(), key=lambda item: item[1]):
        candidates = dataset.frame.loc[dataset.frame["class_name"] == name]
        if len(candidates) < per_class:
            raise ValueError(f"{dataset.split}/{name} 样本不足 {per_class} 张。")
        indices.extend(candidates.sample(n=per_class, random_state=seed + label).index.tolist())
    return sorted(indices)


def get_source_state() -> dict:
    """记录运行时源码哈希和 Git 状态；开发中尚未提交也如实标记。"""
    paths = ("src/model.py", "src/data.py", "src/engine.py", "src/checkpoint.py", "scripts/train.py")
    state = {"source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}}
    try:
        state["git_commit_at_run"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        state["working_tree_dirty_at_run"] = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        state["git_commit_at_run"] = None
        state["working_tree_dirty_at_run"] = None
    return state


def run_training(args, config: dict, run_dir: Path, report_dir: Path) -> None:
    """按明确用途训练并重载最佳模型；无论何种用途，训练过程都只读取 train/val。"""
    purpose = config["purpose"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查驱动/torch 或指定 --device cpu。")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 此阶段使用普通 FP32，关闭 TF32 以便解释和比较重载结果；暂不引入混合精度。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    datasets = {split: FlowerDataset(ROOT, split, Path(config["data_config"])) for split in ("train", "val")}
    indices = {split: (select_smoke_indices(dataset, config[f"{split}_per_class"], config["seed"])
                       if purpose == "smoke" else list(range(len(dataset))))
               for split, dataset in datasets.items()}
    loaders = {
        split: DataLoader(Subset(dataset, indices[split]), batch_size=config["batch_size"],
                          shuffle=split == "train", num_workers=0, drop_last=False,
                          generator=torch.Generator().manual_seed(config["seed"]))
        for split, dataset in datasets.items()
    }
    selected = pd.concat([dataset.frame.iloc[indices[split]] for split, dataset in datasets.items()])
    selected[["relative_path", "class_name", "label", "split", "sha256", "duplicate_group"]].to_csv(
        report_dir / "selected_samples.csv", index=False)
    training_data = datasets["train"]
    source = get_source_state()
    common = {"purpose": purpose, "version": "v0.5", "started_at": datetime.now().astimezone().isoformat(),
              "config": config, "device": str(device), "torch": str(torch.__version__),
              "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
              "num_workers": 0, "cpu_threads": 4, "precision": "FP32, TF32 disabled",
              "train_samples": len(indices["train"]), "val_samples": len(indices["val"]),
              "test_images_used": 0, "manifest_sha256": training_data.manifest_sha256,
              "class_to_idx": training_data.class_to_idx, **source}
    write_json(report_dir / "run.json", {"status": "RUNNING", **common})
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "class_to_idx.json", training_data.class_to_idx)
    model = ResNet18(len(training_data.class_to_idx)).to(device)
    initial_classifier = model.fc.weight.detach().cpu().clone()
    # 交叉熵直接接收 logits 与 int64 标签；SGD 的 momentum 用历史梯度平滑更新。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step_size"],
                                               gamma=config["scheduler_gamma"])
    reference_tensor, reference_label, reference_path = datasets["val"][indices["val"][0]]
    history, best_loss, best_epoch = [], float("inf"), None
    checkpoint_path = run_dir / "best.pt"
    with (report_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as log:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
            buffers = {name: value.clone() for name, value in model.named_buffers()}
            val_metrics = run_epoch(model, loaders["val"], criterion, device)
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
            row = {"purpose": purpose, "epoch": epoch, "learning_rate": learning_rate,
                   **{f"train_{key}": value for key, value in train_metrics.items()},
                   **{f"val_{key}": value for key, value in val_metrics.items()}}
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            log.flush()  # 每轮落盘；中途失败也保留真实进展，不伪造完成状态。
            history.append(row)
            if val_metrics["loss"] < best_loss:
                best_loss, best_epoch = val_metrics["loss"], epoch
                with torch.inference_mode():
                    reference_logits = model(reference_tensor.unsqueeze(0).to(device))[0].cpu().tolist()
                metadata = {"purpose": purpose, "version": "v0.5", "epoch": epoch,
                            "validation_loss": best_loss, "selection_metric": "validation_loss_min",
                            "class_to_idx": training_data.class_to_idx,
                            "preprocessing": training_data.settings,
                            "normalization": training_data.summary["train_rgb_statistics"],
                            "manifest_sha256": training_data.manifest_sha256, "training_config": config,
                            "reference": {"relative_path": reference_path, "label": reference_label,
                                          "logits": reference_logits}, **source}
                save_checkpoint(checkpoint_path, model, metadata)
                write_json(run_dir / "best_metadata.json", metadata)
            print(f"{purpose.upper()} epoch {epoch}/{config['epochs']}: lr={learning_rate:.5g}, "
                  f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.3f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}", flush=True)
            scheduler.step()  # 每轮结束再调学习率；日志记录的是本轮实际使用值。
    classifier_change = float((model.fc.weight.detach().cpu() - initial_classifier).abs().max())
    if classifier_change <= 0:
        raise RuntimeError("分类层参数没有变化，训练流程未通过验收。")
    restored, metadata = load_checkpoint(checkpoint_path, device)
    prediction = check_reference_prediction(restored, metadata, ROOT)
    write_json(report_dir / "reload_prediction.json", prediction)
    summary = {"status": "PASS", **common, "completed_at": datetime.now().astimezone().isoformat(),
               "best_epoch": best_epoch, "best_validation_loss": best_loss,
               "best_validation_accuracy": history[best_epoch - 1]["val_accuracy"],
               "selection_metric": "validation_loss_min", "classifier_max_absolute_change": classifier_change,
               "total_training_batches": sum(row["train_batches"] for row in history),
               "train_loading_seconds": sum(row["train_loading_seconds"] for row in history),
               "train_compute_seconds": sum(row["train_compute_seconds"] for row in history),
               "validation_state_unchanged": True, "reload_prediction_verified": True,
               "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_bytes": checkpoint_path.stat().st_size,
               "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
               "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None,
               "limitations": ("有限样本试运行；非正式成绩；" if purpose == "smoke" else "单次固定种子正式实验；")
                               + "训练阶段未评估测试集；不支持恢复优化器续训"}
    write_json(report_dir / "summary.json", summary)
    write_json(report_dir / "run.json", {"status": "PASS", **common})
    print(f"Best epoch: {best_epoch}; checkpoint: {checkpoint_path}", flush=True)
    print(f"{purpose.upper()} TRAINING PASSED — " +
          ("仅试运行，不是正式实验成绩。" if purpose == "smoke" else "全量训练/验证已完成，测试评估需独立执行。"), flush=True)


def main() -> None:
    """配置文件给出可复现默认值，命令行只覆盖明确指定的参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_smoke.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/v0.4-smoke"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/local/train-smoke"))
    for flag in ("epochs", "batch-size", "seed", "train-per-class", "val-per-class"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    for name in ("epochs", "batch_size", "seed", "train_per_class", "val_per_class", "learning_rate"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    validate_config(config)
    run_dir, report_dir = (ROOT / args.run_dir).resolve(), (ROOT / args.report_dir).resolve()
    if run_dir == report_dir or run_dir.is_relative_to(report_dir) or report_dir.is_relative_to(run_dir):
        raise ValueError("模型目录与报告目录必须分别设置且不能互相包含。")
    for directory in (run_dir, report_dir):
        if not directory.is_relative_to(ROOT):
            raise ValueError("本项目的运行产物须保存于项目目录内部。")
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"拒绝覆盖非空目录：{directory}；重跑请换一个新目录名。")
    for directory in (run_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        run_training(args, config, run_dir, report_dir)
    except Exception as error:
        # 明确留下失败状态；不得因为有 best.pt 或部分日志就声称整次试运行完成。
        write_json(report_dir / "failure.json", {"status": "FAILED", "error_type": type(error).__name__,
                                                  "message": str(error), "config": config})
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, AssertionError) as error:
        print(f"TRAINING FAILED: {error}", file=sys.stderr)
        sys.exit(1)
