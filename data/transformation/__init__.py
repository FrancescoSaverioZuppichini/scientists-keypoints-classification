import numpy as np
from torch import Tensor
import torchvision.transforms as T
from imgaug import augmenters as iaa
from torchvision.transforms.transforms import CenterCrop, RandomCrop

train_transform = T.Compose([T.Normalize(mean=[173.4721, 247.3274], std=[143.8747, 229.3648])])
val_transform = train_transform
