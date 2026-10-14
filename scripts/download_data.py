r"""下载并校验官方花卉数据，保留原图和 LICENSE.txt，不向 Git 添加图片。

运行：.\.venv\Scripts\python.exe scripts\download_data.py
离线复用：添加 --offline；需事先把官方压缩包放入 data/downloads/。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.request
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CHECKSUM_SOURCE = "https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv"
# 2026-10-14 从 TensorFlow Datasets 官方校验清单读取，不能用下载结果反推期望值。
EXPECTED_BYTES = 228813984
EXPECTED_SHA256 = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"


def sha256_file(path):
    """输入文件路径，分块计算 SHA-256；避免将整个数据包读入内存。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_member(member):
    """输入 tar 成员；仅允许 flower_photos 下的普通文件和目录，拒绝越界路径。"""
    path = PurePosixPath(member.name)
    if (path.is_absolute() or not path.parts or path.parts[0] != "flower_photos"
            or ".." in path.parts or "\\" in member.name or ":" in member.name
            or not (member.isfile() or member.isdir())):
        raise ValueError(f"拒绝不安全或不符合预期的压缩包成员：{member.name}")


def verify_extracted(archive, raw_parent):
    """逐个核对已解压文件与已验证压缩包；已有文件不同则报错，不覆盖用户修改。"""
    file_count = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            validate_member(member)
            if not member.isfile():
                continue
            destination = raw_parent / member.name
            if not destination.is_file() or destination.stat().st_size != member.size:
                raise ValueError(f"原始文件缺失或大小变化：{destination}；请保留并移走已有目录后重试。")
            with source.extractfile(member) as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha256_file(destination) != expected:
                raise ValueError(f"原始文件内容变化：{destination}；未覆盖该文件。")
            file_count += 1
    return file_count


def download_and_extract(data_dir, offline=False):
    """输入数据目录和离线选项；获取并校验数据，返回来源记录字典。"""
    data_dir = data_dir.resolve()
    download_dir = data_dir / "downloads"
    raw_parent = data_dir / "raw"
    download_dir.mkdir(parents=True, exist_ok=True)
    raw_parent.mkdir(parents=True, exist_ok=True)
    archive = download_dir / "flower_photos.tgz"
    if not archive.exists():
        if offline:
            raise FileNotFoundError(f"离线模式找不到 {archive}；请先在线下载。")
        partial = archive.with_suffix(".tgz.part")
        print("从 TensorFlow 官方地址下载约 218 MiB，自动沿用已启用的系统代理。", flush=True)
        received = 0
        next_progress = 32 * 1024**2
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, partial.open("wb") as stream:
            while block := response.read(1024**2):
                stream.write(block)
                received += len(block)
                if received >= next_progress:
                    print(f"已下载 {received / 1024**2:.0f} / {EXPECTED_BYTES / 1024**2:.0f} MiB", flush=True)
                    next_progress += 32 * 1024**2
        if received != EXPECTED_BYTES or sha256_file(partial) != EXPECTED_SHA256:
            raise ValueError(f"下载大小或 SHA-256 不符，保留 {partial} 供检查，未解压。")
        partial.rename(archive)
    if archive.stat().st_size != EXPECTED_BYTES or sha256_file(archive) != EXPECTED_SHA256:
        raise ValueError(f"已有压缩包与官方校验值不同：{archive}；未覆盖，请检查来源。")
    print("[PASS] 官方压缩包大小和 SHA-256 校验", flush=True)

    raw_root = raw_parent / "flower_photos"
    if not raw_root.exists():
        # 先解压到新建的工作目录，完成后再改名；中断不会留下假装完整的原始目录。
        staging = data_dir / (".extract-" + uuid.uuid4().hex)
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            for member in members:
                validate_member(member)
            source.extractall(staging, members=members, filter="data")
        (staging / "flower_photos").rename(raw_root)
        staging.rmdir()
    file_count = verify_extracted(archive, raw_parent)
    record = {
        "dataset": "TensorFlow flower_photos / tf_flowers", "source_url": SOURCE_URL,
        "checksum_source": CHECKSUM_SOURCE, "archive_bytes": EXPECTED_BYTES,
        "archive_sha256": EXPECTED_SHA256, "verified_files": file_count,
        "verified_at": datetime.now().astimezone().isoformat(),
        "raw_directory": "raw/flower_photos", "license_file": "raw/flower_photos/LICENSE.txt",
    }
    (data_dir / "source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] {file_count} 个原始文件逐一与压缩包一致")
    print(f"数据目录：{raw_root}\nDATA DOWNLOAD VERIFIED")
    return record


def main():
    """命令行入口；失败时保留已有数据并返回非零退出码。"""
    parser = argparse.ArgumentParser(description="获取官方花卉数据，核对哈希并安全解压。")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--offline", action="store_true", help="只复用并校验已有数据包，不联网。")
    args = parser.parse_args()
    download_and_extract(args.data_dir, args.offline)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"DATA DOWNLOAD FAILED：{error}")
        sys.exit(1)
