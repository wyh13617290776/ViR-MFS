# codes/utils_logger.py
import os
import sys
import logging

def get_logger(name="ViR_MFS", log_file=None, log_level=logging.INFO):
    """
    配置并返回一个标准的 logger。
    支持将日志同时输出到控制台和指定的文件中。
    """
    logger = logging.getLogger(name)
    
    # 防止在多次调用时重复添加 handler 导致日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(log_level)
    
    # 定义日志的输出格式: [时间] [级别] 信息
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler (输出到屏幕)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 文件 Handler (输出到文件)
    if log_file is not None:
        # 确保日志文件所在的目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger