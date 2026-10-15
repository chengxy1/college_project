import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class CleanConsoleFormatter(logging.Formatter):
    """
    控制台专用高颜值格式化器：
    - 去除冗长的文件名与行号噪声
    - 针对多行输出（如表格、分割线）做自动缩进对齐，杜绝错位
    """
    def format(self, record):
        timestamp = self.formatTime(record, "%H:%M:%S")
        prefix = f"[{timestamp}] [{record.levelname:<4}] "
        message = record.getMessage()
        
        # 避免前置换行符导致的空日志前缀
        if message.startswith("\n"):
            message = message.lstrip("\n")
            prefix = "\n" + prefix

        # 如果是多行文本（如表格），每行自动对齐缩进
        lines = message.splitlines()
        if len(lines) > 1:
            first_line = prefix + lines[0]
            indent = " " * len(prefix.lstrip("\n"))
            rest_lines = [indent + line for line in lines[1:]]
            return "\n".join([first_line] + rest_lines)
        return prefix + message

class FileStandardFormatter(logging.Formatter):
    """文件存储专用格式化器：保留完备审计信息"""
    def format(self, record):
        msg = record.getMessage()
        # 清除前导多余换行
        if msg.startswith("\n"):
            record.msg = msg.lstrip("\n")
        return super().format(record)

def setup_logger(log_dir: str = "outputs/logs", log_name: str = "experiment") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    logger = logging.getLogger("EBR_CFR_Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 1. 文件日志 Handler (详细、可追溯)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = FileStandardFormatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    # 2. 控制台 Handler (清爽、无干扰、排版紧凑)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CleanConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"📁 日志系统就绪，记录文件位于: {os.path.abspath(log_file)}")
    return logger


class CheckpointManager:
    """实验断点续传状态机"""
    def __init__(self, checkpoint_dir: str = "outputs/checkpoints", experiment_id: str = "default_eval"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_id}_checkpoint.json")
        self.state = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict[str, Any]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"completed_stages": [], "results": {}, "last_updated": ""}
        return {"completed_stages": [], "results": {}, "last_updated": ""}

    def is_stage_completed(self, stage_name: str) -> bool:
        return stage_name in self.state.get("completed_stages", [])

    def save_stage_result(self, stage_name: str, result_data: Any):
        if stage_name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage_name)
        self.state["results"][stage_name] = result_data
        self.state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def get_stage_result(self, stage_name: str) -> Optional[Any]:
        return self.state.get("results", {}).get(stage_name, None)

    def clear(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            self.state = {"completed_stages": [], "results": {}, "last_updated": ""}
