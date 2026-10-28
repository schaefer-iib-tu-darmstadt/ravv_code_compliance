"""Load the YAML config; exits on a missing or malformed file."""
import yaml
import logging


def load_config(config_path="config/ai_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None: 
            print(f"ERROR: Configuration file '{config_path}' is empty or malformed. Exiting.")
            logging.critical(f"Configuration file '{config_path}' is empty or malformed.")
            exit(1)
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_path}' not found. Exiting.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML configuration file '{config_path}': {e}. Exiting.")
        exit(1)