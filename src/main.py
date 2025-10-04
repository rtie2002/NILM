"""Main entry point for NILM application."""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the NILM application."""
    parser = argparse.ArgumentParser(description='Non-Intrusive Load Monitoring')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'predict'],
                        help='Run mode: train, test, or predict')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    from utils.helpers import setup_logging
    logger = setup_logging()
    
    logger.info(f'Starting NILM in {args.mode} mode')
    
    if args.mode == 'train':
        from models.train import train_model
        train_model(config)
    elif args.mode == 'test':
        from models.test import test_model
        test_model(config)
    elif args.mode == 'predict':
        from models.predict import predict
        predict(config)

if __name__ == '__main__':
    main()
