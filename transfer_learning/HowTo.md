# Extract ResNet101 features for CamVid
python extract_features.py --dataset camvid --backbone resnet101 --batch-size 8

# Extract ResNet101 features for Cityscapes
python extract_features.py --dataset cityscapes --backbone resnet101 --batch-size 4

# Extract VGG16 features (if you want)
python extract_features.py --dataset camvid --backbone vgg16 --batch-size 8
```

This will create a directory structure like:
```
./features/
├── camvid_resnet101/
│   ├── train_00000.pt
│   ├── train_00001.pt
│   ├── ...
│   ├── val_00000.pt
│   ├── ...
│   └── test_00000.pt


# Test ResNet101 features
python feature_dataset.py --feature-dir ./features/camvid_resnet101 --split train

# Test VGG16 features
python feature_dataset.py --feature-dir ./features/camvid_vgg16 --split train


# Test ResNet101 feature dataloaders for CamVid
python create_feature_dataloaders.py \
    --feature-dir ./features/camvid_resnet101 \
    --dataset-info ./CamVid/splits/dataset_info.json \
    --batch-size 16

# Test for Cityscapes
python create_feature_dataloaders.py \
    --feature-dir ./features/cityscapes_resnet101 \
    --dataset-info ./Cityscapes/splits/dataset_info.json \
    --batch-size 8


# Normal training (with backbone)
output = model(images)

# Feature extraction training (without backbone)
output = model.forward_features(x1, x2, x3, x4)  # For ResNet
# or
output = model.forward_features(x3, x4, x5)  # For VGG16


# Step 1: Extract features (run once)
python extract_features.py \
    --dataset camvid \
    --backbone resnet101 \
    --batch-size 8 \
    --output-dir ./features

# Step 2: Train using cached features (much faster!)
python train_fcn_features.py \
    --feature-dir ./features/camvid_resnet101 \
    --dataset-info ./CamVid/splits/dataset_info.json \
    --backbone resnet101 \
    --batch-size 32 \
    --epochs 200 \
    --lr 1e-3

# Resume training
python train_fcn_features.py \
    --feature-dir ./features/camvid_resnet101 \
    --dataset-info ./CamVid/splits/dataset_info.json \
    --backbone resnet101 \
    --resume ./models/FCNs-resnet101_camvid_features_batch32_epoch200_SGD_lr0.001_ReduceLROnPlateau_last.pth
