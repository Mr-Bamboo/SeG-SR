# flake8: noqa
import os.path as osp

import segsr.archs
import segsr.data
import segsr.models
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
