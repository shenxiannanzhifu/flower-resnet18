"""NumPy 分类指标及正式模型选择核对；供 scripts/freeze_model.py 和 evaluate.py 使用。

不训练或调参。混淆矩阵约定：行是真实类别，列是预测类别。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_json_sha256(content: bytes) -> str:
    """对 JSON 内容计算稳定摘要；输入 UTF-8 字节，输出 SHA-256 十六进制字符串。

    JSON 的换行、缩进和键顺序不是实验内容。先解析再按固定规则编码，
    避免 Windows CRLF 被 Git 转成 LF 后，将同一份报告误判为被修改。
    模型权重仍按原始二进制字节校验，不使用此函数。
    """
    value = json.loads(content.decode("utf-8"))
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def training_summary_matches(content: bytes, selection: dict) -> bool:
    """按选择记录声明的格式核对训练报告，兼容早期未声明格式的记录。

    历史记录不能在测试后重写：旧格式仅容许 Git 的 LF/CRLF 转换，
    不容许改数值、缩进或其他内容；新记录明确采用 canonical-json-v1。
    """
    format_name = selection.get("training_summary_hash_format", "raw-bytes-v1")
    expected = selection["training_summary_sha256"]
    if format_name == "canonical-json-v1":
        return canonical_json_sha256(content) == expected
    if format_name == "raw-bytes-v1":
        lf = content.replace(b"\r\n", b"\n")
        candidates = (content, lf, lf.replace(b"\n", b"\r\n"))
        return any(hashlib.sha256(value).hexdigest() == expected for value in candidates)
    raise ValueError(f"未知训练报告哈希格式：{format_name}")


def classification_metrics(labels, predictions, class_names: list[str]) -> dict:
    """输入同长度整数类别序列，返回混淆矩阵、准确率和逐类/宏平均/加权指标。

    precision=TP/预测为该类数，recall=TP/真实该类数；零分母明确定义为 0。
    macro 对所有类等权，weighted 按真实支持数加权，不掩盖少数类表现。
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    count = len(class_names)
    if (count == 0 or len(set(class_names)) != count or labels.ndim != 1
            or predictions.shape != labels.shape or labels.size == 0):
        raise ValueError("类别须非空且唯一，标签与预测须为等长非空一维数组。")
    for values in (labels, predictions):
        if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= count):
            raise ValueError("标签/预测必须是合法的整数类别编号。")
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support, predicted_count = matrix.sum(axis=1), matrix.sum(axis=0)
    true_positive = matrix.diagonal().astype(float)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(count), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros(count), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count), where=precision + recall > 0)
    return {"samples": int(support.sum()), "correct": int(true_positive.sum()),
            "accuracy": float(true_positive.sum() / support.sum()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support)),
            "confusion_matrix": matrix.tolist(), "matrix_axes": "rows=true, columns=predicted",
            "zero_division": 0,
            "per_class": [{"class_name": name, "support": int(support[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, name in enumerate(class_names)]}


def verified_training_summary(checkpoint: Path, report_dir: Path) -> dict:
    """从实际逐轮日志独立复核最优轮，确保选择依据是完整验证集的最小损失。"""
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    if (summary["status"] != "PASS" or summary["purpose"] != "formal"
            or summary["test_images_used"] != 0 or summary["selection_metric"] != "validation_loss_min"):
        raise ValueError("只能冻结已完成、未评估测试集且按验证损失选出的正式模型。")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != summary["checkpoint_sha256"]:
        raise ValueError("模型文件哈希与正式训练报告不一致。")
    history = pd.read_csv(report_dir / "metrics.csv")
    if (len(history) != summary["config"]["epochs"] or not (history.purpose == "formal").all()
            or list(history.epoch) != list(range(1, len(history) + 1))
            or not np.isfinite(history.val_loss).all()
            or not (history.train_samples == summary["train_samples"]).all()
            or not (history.val_samples == summary["val_samples"]).all()):
        raise ValueError("正式训练日志不完整、样本数不一致或验证损失无效。")
    best = history.loc[history.val_loss.idxmin()]
    if int(best.epoch) != summary["best_epoch"] or not np.isclose(best.val_loss, summary["best_validation_loss"]):
        raise ValueError("保存模型并非逐轮日志中验证损失最小的轮次。")
    return summary
