# flake8: noqa
import os.path as osp

from basicsr import archs,data,losses,models,metrics

from basicsr.test import test_pipeline
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)
