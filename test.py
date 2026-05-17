"""Compatibility entrypoint for ViR-MFS testing."""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ViR-MFS with YAML configuration files.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to project configuration YAML.")
    parser.add_argument("--params", default="config/params.yaml", help="Path to parameter configuration YAML.")
    args = parser.parse_args()

    from engine.testing import test_model

    test_model(config_path=args.config, params_path=args.params)
