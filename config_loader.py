# codes/config_loader.py
import os
import yaml

def load_configs(config_path="config/config.yaml", params_path="config/params.yaml"):
    """
    加载核心配置文件和超参数配置文件。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Error] 找不到核心配置文件: {config_path}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"[Error] 找不到超参数配置文件: {params_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    return cfg, params