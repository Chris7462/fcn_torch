"""
Complete CamVid Dataset Usage Example
Shows how to use the dataset and dataloaders for training
"""

import torch
import torch.nn as nn
from create_dataloaders import create_dataloaders


def main():
    # Step 1: Configure paths (adjust these to your actual paths)
    RAW_IMAGE_DIR = '/data/CamVid/701_StillsRaw_full'
    LABEL_DIR = '/data/CamVid/LabeledApproved_full'
    SPLITS_DIR = '/data/CamVid/splits'
    DATASET_INFO_PATH = '/data/CamVid/splits/dataset_info.json'

    # Step 2: Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        raw_image_dir=RAW_IMAGE_DIR,
        label_dir=LABEL_DIR,
        splits_dir=SPLITS_DIR,
        dataset_info_path=DATASET_INFO_PATH,
        batch_size=8,
        num_workers=4,
        target_size=(480, 360),
        use_computed_stats=True
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    class_weights = dataloaders['class_weights']
    num_classes = dataloaders['num_classes']
    ignore_index = dataloaders['ignore_index']

    # Step 3: Setup loss function with class weights and ignore_index
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=ignore_index
    )

    print(f"\nLoss function configured:")
    print(f"  Using class weights: {class_weights.shape}")
    print(f"  Ignore index: {ignore_index}")

    # Step 4: Example training loop structure
    print("\nExample training loop structure:")
    print("=" * 60)

    # Pseudo model (replace with your actual segmentation model)
    # model = YourSegmentationModel(num_classes=num_classes)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop example
    num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        # model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)  # Shape: [B, 3, H, W]
            masks = batch['mask'].to(device)    # Shape: [B, H, W]

            # Forward pass
            # outputs = model(images)  # Shape: [B, num_classes, H, W]
            # loss = criterion(outputs, masks)

            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # train_loss += loss.item()

            if batch_idx == 0:
                print(f"  Batch shape - Images: {images.shape}, Masks: {masks.shape}")

        # Validation phase
        # model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # outputs = model(images)
                # loss = criterion(outputs, masks)
                # val_loss += loss.item()

        # Print epoch stats
        # avg_train_loss = train_loss / len(train_loader)
        # avg_val_loss = val_loss / len(val_loader)
        # print(f"  Train Loss: {avg_train_loss:.4f}")
        # print(f"  Val Loss:   {avg_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Setup complete! Ready for training.")
    print("\nTo use in your training script:")
    print("1. Replace pseudo model with your actual segmentation model")
    print("2. Uncomment the training/validation code")
    print("3. Add metrics computation (IoU, mIoU, etc.)")
    print("4. Add model checkpointing")


if __name__ == '__main__':
    main()
