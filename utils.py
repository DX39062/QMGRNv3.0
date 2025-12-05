import os
import logging
from datetime import datetime

def setup_logger(model_name, extra_info=None):
    """
    配置日志
    :param model_name: 模型名称 (lstm, qlstm, gcn_only)
    :param extra_info: 额外的文件名后缀字符串 (例如 "_bias0.2")，如果为 None 则不添加
    """
    # 确保日志目录存在
    os.makedirs("./logging", exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件名
    if extra_info:
        # 确保 extra_info 以 _ 开头，或者这里统一加 _
        if not extra_info.startswith("_"):
            extra_info = "_" + extra_info
        log_filename = f"./logging/train_{model_name}{extra_info}_{timestamp}.txt"
    else:
        log_filename = f"./logging/train_{model_name}_{timestamp}.txt"
    
    # 配置 logging
    # 如果已经有 handler，先清除，防止重复打印
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers = []

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename), # 输出到文件
            logging.StreamHandler()            # 输出到控制台
        ]
    )
    return log_filename