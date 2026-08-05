# Copyright (c) ModelScope Contributors. All rights reserved.
from .default_flow import DefaultFlowCallback
from .print import PrintCallback
from .repro_receipt import ReproReceiptCallback
from .swanlab import SwanlabCallback
from .tensorboard import TensorboardCallback
from .wandb import WandbCallback

megatron_callbacks_map = {
    'print': PrintCallback,
    'repro_receipt': ReproReceiptCallback,
    'default_flow': DefaultFlowCallback,
    'swanlab': SwanlabCallback,
    'wandb': WandbCallback,
    'tensorboard': TensorboardCallback,
}
