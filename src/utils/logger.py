import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name="AgentLN"):
    # 确保根目录下有 logs 文件夹
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 格式化: [时间] - [级别] - [文件名:行号] - 信息
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # 1. 输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 2. 输出到文件 (每天一个文件，最多保留 7 天，单个文件最大 10MB)
        log_file = os.path.join(log_dir, "app.log")
        file_handler = TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# 实例化全局单例 logger
logger = setup_logger()