"""训练/验证一轮的基础循环；供 scripts/train.py 调用。

运行项目入口：.venv/Scripts/python.exe scripts/train.py --device cuda
这里负责前向、损失、反向传播和按样本加权的指标，不负责数据划分或保存文件。
"""

import time

import torch


def synchronize(device: torch.device) -> None:
    """GPU 运算异步执行；计时前等待完成，避免把未完成的计算误算成很快。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, criterion, device: torch.device, optimizer=None) -> dict:
    """遍历一个子集，返回平均损失、准确率、样本数及诊断耗时。

    loader 每项为 (images, labels, paths)。传入 optimizer 表示训练，否则只做验证。
    criterion 须按批次输出平均损失；本项目使用默认 reduction='mean' 的交叉熵。
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples, batches = 0.0, 0, 0, 0
    loading_seconds, compute_seconds = 0.0, 0.0
    synchronize(device)
    start = time.perf_counter()
    last_end = start
    # eval() 控制 BN 行为；inference_mode() 关闭求导，二者不能互相替代。
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, _paths in loader:
            compute_start = time.perf_counter()
            loading_seconds += compute_start - last_end
            images, labels = images.to(device), labels.to(device)
            if training:
                # 每个批次先清除旧梯度，避免默认的梯度累加改变更新含义。
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("出现 NaN/Inf 损失，停止运行；不要把无效结果当作训练成功。")
            if training:
                loss.backward()  # 自动求导：沿计算图计算每个参数的梯度。
                optimizer.step()  # 优化器根据梯度和学习率更新权重。
            count = len(labels)
            # 最后一批可能较小，按样本数加权，不能直接平均各批次的均值。
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum())
            total_samples += count
            batches += 1
            synchronize(device)
            last_end = time.perf_counter()
            compute_seconds += last_end - compute_start
    if total_samples == 0:
        raise ValueError("当前数据加载器没有样本，无法计算损失或准确率。")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "samples": total_samples, "batches": batches,
            "loading_seconds": loading_seconds, "compute_seconds": compute_seconds,
            "elapsed_seconds": time.perf_counter() - start}
