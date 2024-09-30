from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SIRSTDataset(CustomDataset):


    CLASSES=('background', 'front')
    PALETTE=[[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(SIRSTDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)