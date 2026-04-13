import yaml
from deepmerge import always_merger
from pkg_credit_default.utils.logger import logger
from pkg_credit_default.config.config_paths import BASE_CONFIG_PATH, ENV_CONFIG_PATH

def _load_yaml(path):
    logger.info(f"load_yaml() - Loading YAML configuration from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _load_config():
    logger.info("load_config() - Starting to load configuration...")
    base = _load_yaml(BASE_CONFIG_PATH)
    env  = _load_yaml(ENV_CONFIG_PATH)

    logger.info("load_config() - Configuration loaded successfully.")
    return always_merger.merge(base, env)

CONFIG = _load_config()