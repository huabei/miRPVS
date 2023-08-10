"""
@author:ZNDX
@file:zinc_complex3a6p_data.py
@time:2022/10/13
"""

from .zinc_complex3a6p_data_300k import ZincComplex3a6pData300k


class ZincComplex4z4cData300k(ZincComplex3a6pData300k):
    def __init__(self, data_dir, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(data_dir, train, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """返回原始数据文件名."""
        return ["4z4c_1m.h5"]
