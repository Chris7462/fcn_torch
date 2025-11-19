"""
CamVid Dataset Training Configuration
11 classes for semantic segmentation
"""

# Model configuration (to be implemented in Phase 2)
net = dict(
    type='FCNs',
)

backbone = dict(
    type='VGG16',
    pretrained=True,
)

# Optimizer configuration
optimizer = dict(
    type='sgd',
    lr=0.001,
    weight_decay=5e-4,
    momentum=0.9
)

# Learning rate scheduler
scheduler = dict(
    type='StepLR',
    step_size=50,
    gamma=0.5
)

# Image settings
img_height = 352
img_width = 480

# Note: mean and std are loaded from dataset_info.json
# These values are computed by tools/prepare_camvid.py
img_norm = dict(
    mean=[0.485, 0.456, 0.406],  # Will be overridden by dataset_info.json
    std=[0.229, 0.224, 0.225]     # Will be overridden by dataset_info.json
)

# Dataset paths
dataset_root = './CamVid'
dataset = dict(
    train=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/train.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
    ),
    val=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/val.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
    ),
    test=dict(
        type='CamVid',
        img_dir='./data/CamVid/701_StillsRaw_full',
        label_dir='./data/CamVid/LabeledApproved_full',
        split_file='./data/CamVid/splits/test.txt',
        dataset_info_path='./data/CamVid/splits/dataset_info.json',
    )
)

# Training hyperparameters
batch_size = 16
workers = 4
num_classes = 11
ignore_label = 255
epochs = 200

# Logging
save_ep = 10  # Save checkpoint every N epochs

# Output directories
model_dir = './models'
plot_dir = './plots'
