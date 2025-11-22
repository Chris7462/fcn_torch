"""
Extract and cache backbone features for faster training
Supports both VGG16 and ResNet101 backbones
"""

import torch
import os
import argparse
from tqdm import tqdm
from create_camvid_dataloaders import create_camvid_dataloaders
from create_cityscapes_dataloaders import create_cityscapes_dataloaders
from fcn import create_fcn_model


def extract_features(dataloader, model, device, output_dir, split_name, backbone='vgg16'):
    """
    Extract and save features from backbone

    Args:
        dataloader: PyTorch DataLoader
        model: FCN model with backbone
        device: torch device
        output_dir: Directory to save features
        split_name: 'train', 'val', or 'test'
        backbone: 'vgg16' or 'resnet101'
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    feature_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {split_name}")):
            images = batch['image'].to(device)
            filenames = batch['filename']
            masks = batch['mask']

            # Extract features from backbone
            features = model.pretrained_net(images)

            # Save features for each image in batch
            for i in range(len(filenames)):
                if backbone == 'vgg16':
                    # VGG16: 3 feature maps
                    feature_dict = {
                            'x3': features['x3'][i].cpu(),  # (256, H/8, W/8)
                            'x4': features['x4'][i].cpu(),  # (512, H/16, W/16)
                            'x5': features['x5'][i].cpu(),  # (512, H/32, W/32)
                            'mask': masks[i],
                            'filename': filenames[i]
                            }
                elif backbone == 'resnet101':
                    # ResNet101: 4 feature maps
                    feature_dict = {
                            'x1': features['x1'][i].cpu(),  # (256, H/4, W/4)
                            'x2': features['x2'][i].cpu(),  # (512, H/8, W/8)
                            'x3': features['x3'][i].cpu(),  # (1024, H/16, W/16)
                            'x4': features['x4'][i].cpu(),  # (2048, H/32, W/32)
                            'mask': masks[i],
                            'filename': filenames[i]
                            }
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")

                # Save to disk
                save_path = os.path.join(output_dir, f"{split_name}_{feature_count:05d}.pt")
                torch.save(feature_dict, save_path)
                feature_count += 1

    print(f"✓ {split_name} features saved to {output_dir}")
    print(f"  Total features extracted: {feature_count}")


def main():
    parser = argparse.ArgumentParser(description='Extract and cache backbone features')
    parser.add_argument('--dataset', type=str, default='camvid', choices=['camvid', 'cityscapes'],
                        help='Dataset to use (default: camvid)')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet101'],
                        help='Backbone architecture (default: resnet101)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for feature extraction (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./features',
                        help='Directory to save extracted features (default: ./features)')
    args = parser.parse_args()

    # Dataset configurations
    DATASET_CONFIGS = {
            'camvid': {
                'raw_image_dir': './CamVid/701_StillsRaw_full',
                'label_dir': './CamVid/LabeledApproved_full',
                'splits_dir': './CamVid/splits',
                'dataset_info_path': './CamVid/splits/dataset_info.json',
                'target_size': (480, 352),
                },
            'cityscapes': {
                'leftimg_dir': './Cityscapes/leftImg8bit',
                'gtfine_dir': './Cityscapes/gtFine',
                'splits_dir': './Cityscapes/splits',
                'dataset_info_path': './Cityscapes/splits/dataset_info.json',
                'target_size': (2048, 1024),
                }
            }

    config = DATASET_CONFIGS[args.dataset]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    if args.dataset == 'camvid':
        dataloaders = create_camvid_dataloaders(
                raw_image_dir=config['raw_image_dir'],
                label_dir=config['label_dir'],
                splits_dir=config['splits_dir'],
                dataset_info_path=config['dataset_info_path'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                target_size=config['target_size']
                )
    elif args.dataset == 'cityscapes':
        dataloaders = create_cityscapes_dataloaders(
                leftimg_dir=config['leftimg_dir'],
                gtfine_dir=config['gtfine_dir'],
                splits_dir=config['splits_dir'],
                dataset_info_path=config['dataset_info_path'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                target_size=config['target_size']
                )

    num_classes = dataloaders['num_classes']

    # Create model with frozen backbone
    print(f"\nCreating FCN model with {args.backbone} backbone...")
    model = create_fcn_model(n_class=num_classes, backbone=args.backbone, pretrained=True)
    model = model.to(device)
    model.eval()

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{args.backbone}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Extract features for all splits
    print("\n" + "="*80)
    print("Starting Feature Extraction")
    print("="*80)

    for split_name in ['train', 'val', 'test']:
        print(f"\nExtracting {split_name} features...")
        extract_features(
                dataloader=dataloaders[split_name],
                model=model,
                device=device,
                output_dir=output_dir,
                split_name=split_name,
                backbone=args.backbone
                )

    print("\n" + "="*80)
    print("Feature Extraction Complete!")
    print("="*80)
    print(f"\nFeatures saved to: {output_dir}")
    print(f"You can now train using: python train_fcn_features.py --feature-dir {output_dir}")


if __name__ == '__main__':
    main()
