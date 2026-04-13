import os
from pathlib import Path
from pkg_credit_default.config.settings import settings


# define the environment variable to determine which config to use
ENV = settings.ENV  

# define the paths to the config files - shared config between dev and prod.
BASE_CONFIG_PATH = Path("configs/base/base.yaml")

# define the path to the environment-specific config file.
ENV_CONFIG_PATH = Path(f"configs/{ENV}/training.yaml")
