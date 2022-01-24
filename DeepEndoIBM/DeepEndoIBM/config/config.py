# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:17:45 2021

@author: cypri
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfig:
    train_path: str
    val_path: str
    ers_test_path: str
    ers_precise_test_path: str
    ers_imprecise_test_path: str
    hyperkvasir_test_path: str
    model_path: str = "./"
    save_model: bool = True
    generate_plots: bool = True
    generate_metrics: bool = True
    generate_samples: bool = True
    n_samples: int = 5

@dataclass
class SegmentationConfig:
    model_name: str = ""
    model_config: Dict = field(default_factory=Dict)
    cross_validation: bool = True
    cv_split: int = 5
    loss_function: str = ""
    augmentations: Dict = field(default_factory=Dict)
    metrics: List = field(default_factory=List)
    optimizer: str = ""
    batch_size: int = 10
    target_size: int = 128
    val_split: float = 0.1
    test_split: float = 0.1
    train_patience: int = 15
    monitor_value: str = "val_loss"
    epochs: int = 150


class LocalizationConfig:
    augmentations: List = field(default_factory=List)
    model_name: str = ""
    model_config: Dict = field(default_factory=Dict)
    metrics: List = field(default_factory=List)
    optimizer: str = ""
    train_classifier: bool = False
    transfer_learning: bool = False
    precise_labels: bool = False
    batch_size: int = 10
    target_size: int = 128
    val_split: float = 0.1
    test_split: float = 0.1
    train_patience: int = 15
    epochs: int = 150  
    localization_head_layers: List = field(default_factory=List)
    classification_head_layers: List = field(default_factory=List)
