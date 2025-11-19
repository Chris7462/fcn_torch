"""
Main Entry Point for FCN Training/Validation/Testing
"""

import argparse
import torch.backends.cudnn as cudnn

from utils.config import Config
from runner.runner import Runner
from datasets import build_dataloader


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    print(f"Loaded config from: {args.config}")

    # Update config with args
    cfg.load_from = args.load_from
    cfg.work_dirs = args.work_dirs
    cfg.test = args.test
    cfg.validate = args.validate

    # Set cudnn
    cudnn.benchmark = True
    cudnn.fastest = True

    # Create runner
    runner = Runner(cfg)

    # Run test, validation, or training
    if args.test:
        # Test mode - run inference and save predictions
        test_loader = build_dataloader(cfg.dataset.test, cfg, is_train=False)
        runner.test(test_loader)
    elif args.validate:
        # Validation only mode
        val_loader = build_dataloader(cfg.dataset.val, cfg, is_train=False)
        avg_loss, mean_iou, mean_pixel_acc, class_ious = runner.validate(val_loader)
        runner.recorder.logger.info("\n" + "="*80)
        runner.recorder.logger.info("Validation Results")
        runner.recorder.logger.info("="*80)
        runner.recorder.logger.info(f"Val Loss:       {avg_loss:.4f}")
        runner.recorder.logger.info(f"Val mIoU:       {mean_iou:.4f}")
        runner.recorder.logger.info(f"Val Pixel Acc:  {mean_pixel_acc:.4f}")
        runner.recorder.logger.info("="*80)
    else:
        # Training mode (default)
        runner.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train/Test FCN model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument(
        '--work_dirs',
        type=str,
        default='work_dirs',
        help='Work directory for saving logs and checkpoints'
    )
    parser.add_argument(
        '--load_from',
        default=None,
        help='Path to checkpoint file to resume from'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation only (requires --load_from)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run testing and save predictions (requires --load_from)'
    )
    args = parser.parse_args()

    # Validation
    if (args.validate or args.test) and not args.load_from:
        parser.error('--validate and --test require --load_from to specify checkpoint')

    return args


if __name__ == '__main__':
    main()
