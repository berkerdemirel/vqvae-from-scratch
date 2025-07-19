import warnings

import torch

# 1. Use H100 Tensor Cores for fp32 matmul
torch.set_float32_matmul_precision("high")

# 2. Silence torchvision’s VGG16 weight deprecation spam
warnings.filterwarnings(
    "ignore",
    message=r"The parameter 'pretrained' is deprecated",  # regex OK
    module="torchvision",
)
warnings.filterwarnings(
    "ignore",
    message=r"Arguments other than.*for 'weights' are deprecated",
    module="torchvision",
)

warnings.filterwarnings(
    "ignore",
    message=r"No device id is provided via `init_process_group` or `barrier `.*",
    category=UserWarning,
    module="torch.distributed.distributed_c10d",
)
