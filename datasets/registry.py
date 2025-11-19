from utils import Registry, build_from_cfg
import torch


DATASETS = Registry('datasets')


def build_dataset(split_cfg, cfg):
    """
    Build dataset from config

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        cfg: Global config object

    Returns:
        Dataset instance
    """
    args = split_cfg.copy()
    args.pop('type')
    args = args.to_dict() if hasattr(args, 'to_dict') else dict(args)
    args['cfg'] = cfg
    return build_from_cfg(split_cfg, DATASETS, default_args=args)


def build_dataloader(split_cfg, cfg, is_train=True):
    """
    Build PyTorch DataLoader from config

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        cfg: Global config object
        is_train: Whether this is training data (affects shuffle)

    Returns:
        DataLoader instance
    """
    dataset = build_dataset(split_cfg, cfg)

    shuffle = is_train

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False
    )

    return data_loader
