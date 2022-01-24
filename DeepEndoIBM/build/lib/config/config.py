# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:17:45 2021

@author: cypri
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PipelineConfig:
    augmentations: List = field(default_factory=List)
    mode: Dict = field(default_factory=Dict)
    metrics: List = field(default_factory=List)
    optimizer: str
    batch_size: int = 10
    target_size: int = 128
    training_split: int = 0.8

  