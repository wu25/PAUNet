import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data as data
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment, paired_random_crop,random_augmentation


@DATASET_REGISTRY.register()
class TrainDataset(Dataset):
    # def __init__(self, data_dir, patch=256):
    def __init__(self, opt):
        super(TrainDataset, self).__init__()
        self.opt = opt
        self.noise_fns = []
        self.clean_fns = []
        self.noise_fns_2 = []
        self.clean_fns_2 = []
        self.data_dir = opt['data_dir']
        self.patch = opt['patch']
        self.noise_fns = glob.glob(os.path.join(self.data_dir + "/input" + "/**"))
        self.clean_fns = glob.glob(os.path.join(self.data_dir + "/groundtruth" + "/**"))
        # self.noise_fns = glob.glob(os.path.join(self.data_dir  +"/**/*"+"NOISY_SRGB_010.PNG"))
        # self.clean_fns = glob.glob(os.path.join(self.data_dir + "/**/*"+"GT_SRGB_010.PNG"))
        self.noise_fns.sort()
        self.clean_fns.sort()
        print('fetch {} samples for training'.format(len(self.noise_fns)))

    def __getitem__(self, index):
        # fetch image
        flag_aug = random.randint(0, 2)
        fn_noisy = self.noise_fns[index]
        # print("noisy:",fn_noisy)
        noisy = Image.open(fn_noisy).convert('RGB')  # 加了RGB
        noisy = np.array(noisy, dtype=np.float32)
        if (flag_aug == 0):
            transformer = transforms.Compose([ transforms.ToTensor(), transforms.RandomHorizontalFlip(1)])
        elif (flag_aug == 1):
            transformer = transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(1)])
        else:
            transformer = transforms.Compose([ transforms.ToTensor()])
        # transformer = transforms.Compose([transforms.ToTensor()])
        noisy = transformer(noisy)

        fn_clean = self.clean_fns[index]
        clean = Image.open(fn_clean).convert('RGB')  # 加了RGB
        clean = np.array(clean, dtype=np.float32)
        clean = transformer(clean)
        # noisy,clean=random_cut(image_noisy=noisy,image_gt=clean,w=self.patch)
        # return noisy,clean
        clean = clean / 255.0
        noisy = noisy / 255.0
        # print(torch.max(clean))
        return {'lq': noisy, 'lq_path': fn_noisy, 'gt': clean, 'gt_path': fn_noisy}

    def __len__(self):
        return len(self.noise_fns)


@DATASET_REGISTRY.register()
class ValDataset(Dataset):
    # def __init__(self, data_dir):
    def __init__(self, opt):
        super(ValDataset, self).__init__()
        self.opt = opt
        self.noise_fns = []
        self.data_dir = opt['data_dir']
        self.noise_fns = glob.glob(os.path.join(self.data_dir + "/input" + "/**"))
        self.clean_fns = glob.glob(os.path.join(self.data_dir + "/input" + "/**"))
        self.noise_fns.sort()
        self.clean_fns.sort()
        print('fetch {} samples for testing'.format(len(self.noise_fns)))

    def __getitem__(self, index):
        # fetch image
        fn_noisy = self.noise_fns[index]
        # print("noisy:",fn_noisy)
        noisy = Image.open(fn_noisy).convert()  # 加了RGB
        noisy = np.array(noisy, dtype=np.float32)
        transformer = transforms.Compose([transforms.ToTensor()])
        # transformer = transforms.Compose([transforms.ToTensor()])
        noisy = transformer(noisy)

        fn_clean = self.clean_fns[index]
        clean = Image.open(fn_clean).convert()  # 加了RGB
        clean = np.array(clean, dtype=np.float32)
        clean = transformer(clean)

        clean = clean / 255.0
        noisy = noisy / 255.0
        return {'lq': noisy, 'lq_path': fn_noisy, 'gt': clean, 'gt_path': fn_noisy}

    def __len__(self):
        return len(self.noise_fns)

@DATASET_REGISTRY.register()
class Dataset_MIRNETV2(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_MIRNETV2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        self.filename_tmpl = '{}'


        self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            # # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
            #                                     gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        print(img_gt)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class Dataset_PairedImage(Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        # self.paths = paired_paths_from_folder(
        #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #     self.filename_tmpl)
        # self.gt_paths=glob.glob(os.path.join(self.gt_folder + "/**"))
        # self.lq_paths=glob.glob(os.path.join(self.lq_folder + "/**"))
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(self.gt_paths[index]))
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(self.lq_paths[index]))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                self.gt_paths[index])

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': self.lq_paths[index],
            'gt_path': self.gt_paths[index]
        }

    def __len__(self):
        return len(self.gt_paths)


@DATASET_REGISTRY.register()
class Dataset_GaussianDenoising(Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.gt_paths = glob.glob(os.path.join(self.gt_folder + "/**"))
        self.gt_paths.sort()
        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']
            print('fetch {} samples for training'.format(len(self.gt_paths)))
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # index = index % len(self.gt_paths)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
                # img_gt = Image.open(gt_path).convert('RGB')  # 加了RGB
                # img_gt = np.array(img_gt, dtype=np.float32)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
                # img_gt = Image.open(gt_path).convert('RGB')  # 加了RGB
                # img_gt = np.array(img_gt, dtype=np.float32)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)
            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)
            # noise_level = torch.FloatTensor([self.sigma_test]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            # noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            # img_lq.add_(noise)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        # return len(self.paths)
        return len(self.gt_paths)




