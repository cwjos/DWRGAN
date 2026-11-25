# #-*-coding:utf-8-*-
#
from pathlib import Path
from itertools import chain
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from munch import Munch
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
# 为了生成不成对的训练数据，子集被随机分为三个分区：
# 1）第一个分区有 2,250 张原始照片作为低质量输入；
# 2）第二个分区由另外2,250张原始照片的修饰版本组成，作为所需的高质量照片；
# 3) 最后一个分区是剩余的 500 张原始照片，用于验证（100 张图像）和测试（400 张图像）。
# 这三个部分彼此没有重叠。

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        print(f"Image before transform: {type(img)}")
        if self.transform is not None:
            img = self.transform(img)
        print(f"Image after transform: {type(img)}")  # 输出图像类型
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        # print(f"Found directories: {domains}")
        fnames, fnames2 = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            # print(f"Found {len(cls_fnames)} images in {domain}")
            if idx == 0:
                fnames += cls_fnames
            elif idx == 1:
                fnames2 += cls_fnames
        # pairs = list(zip(fnames, fnames2))
        # print(f"Created fnames {len(fnames)} image pairs")
        # print(f"Created fnames2 {len(fnames2)} image pairs")
        # return pairs
        return list(zip(fnames, fnames2))

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        name = str(fname2)
        img_name, _ = name.split('.', 1)
        _, img_name = img_name.rsplit('\\', 1)  # 对于 Unix 系统，换成 '/' 作为路径分隔符
        #先加载为PIL Image
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        # 如果 transform 被提供，则应用 transform
        if self.transform is not None:
            # print(f"before image type:{type(img)} | {type(img2)}")
            img = self.transform(img)
            img2 = self.transform(img2)
            # print(f"after image type:{type(img)} | {type(img2)}")
        return img, img2, img_name

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, img_size=400, resize_size=256, batch_size=8, shuffle=True, num_workers=8, drop_last=True):
    print(f"\n数据加载配置:")
    print(f"数据集路径: {root}")
    print(f"图片大小: {img_size}")
    print(f"调整大小: {resize_size}")
    print(f"批次大小: {batch_size}")
    transform = transforms.Compose([
        transforms.RandomCrop(img_size),
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)
    print(f"数据集大小: {len(dataset)}")
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=1000, batch_size=4, shuffle=False, num_workers=4):
    transform = transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        # transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ReferenceDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        print(f"初始化 InputFetcher，数据加载器大小: {len(loader)}")
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.iter = iter(self.loader)

    def _fetch_refs(self):
        try:
            x, y, name = next(self.iter)
            # print(f"加载数据: {name}") #如果需要查看每次加载的图片名称
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y, name = next(self.iter)
        return x, y, name

    def __next__(self):
        x, y, img_name = self._fetch_refs()
        x, y = x.to(self.device), y.to(self.device)
        inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)

        return inputs


# from pathlib import Path
# from itertools import chain
# import os
# import random
#
# from munch import Munch
# from PIL import Image
# import torch
# from torch.utils import data
# from torchvision import transforms
# from torchvision.transforms import functional as TF
#
#
# def listdir(dname):
#     exts = ['png', 'jpg', 'jpeg', 'JPG']
#     return sorted(chain(*[list(Path(dname).rglob(f'*.{ext}')) for ext in exts]))
#
#
# # ------------------------------
# # 1) 训练集（非成对）：trainA 和 trainB 不一一对应
# # ------------------------------
# class UnpairedTrainDataset(data.Dataset):
#     """root 下有两个子目录：trainB/ 与 trainA/（字母序第一个视为 trainB，第二个视为 trainA）。
#     训练时各自独立增广；每步取 trainA[i] 与 随机 trainB[j]（非成对）。"""
#     def __init__(self, root, img_size=512, crop_size=256):
#         subdirs = sorted(os.listdir(root))
#         assert len(subdirs) >= 2, f"Expect 2 sub-dirs under {root}, got: {subdirs}"
#         exp_dir = os.path.join(root, subdirs[0])  # 第一个目录=trainB（高质）
#         raw_dir = os.path.join(root, subdirs[1])  # 第二个目录=trainA（低质）
#
#         self.exp_list = [str(p) for p in listdir(exp_dir)]
#         self.raw_list = [str(p) for p in listdir(raw_dir)]
#         assert len(self.exp_list) and len(self.raw_list), "trainB/trainA 文件夹不能为空"
#
#         self.img_size = img_size
#         self.crop_size = crop_size
#         self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#
#     def __len__(self):
#         return max(len(self.raw_list), len(self.exp_list))
#
#     def _rand_crop_flip(self, pil):
#         pil = transforms.Resize(self.img_size)(pil)
#         i, j, h, w = transforms.RandomCrop.get_params(pil, (self.crop_size, self.crop_size))
#         pil = TF.crop(pil, i, j, h, w)
#         if torch.rand(1) < 0.5:
#             pil = TF.hflip(pil)
#         if torch.rand(1) < 0.5:
#             pil = TF.vflip(pil)
#         ten = TF.to_tensor(pil)
#         return self.normalize(ten)
#
#     def __getitem__(self, idx):
#         raw_path = self.raw_list[idx % len(self.raw_list)]
#         exp_path = random.choice(self.exp_list)
#
#         img_raw = Image.open(raw_path).convert('RGB')
#         img_exp = Image.open(exp_path).convert('RGB')
#
#         # 非成对训练：各自独立增广
#         img_raw = self._rand_crop_flip(img_raw)
#         img_exp = self._rand_crop_flip(img_exp)
#
#         name = Path(raw_path).stem  # 用 trainA 的名字作为样本名
#         return img_exp, img_raw, name
#
#
# # ------------------------------
# # 2) 评测集（成对）：按文件名一一配对（例如 val/ 与 test/）
# # ------------------------------
# class PairedEvalDataset(data.Dataset):
#     """root 下子目录包含 trainA 与 trainB（或 label/gt/target/expert 等）。按文件名（不含扩展名）一一配对用于评测。"""
#     def __init__(self, root, img_size=512):
#         sub = {d: os.path.join(root, d) for d in os.listdir(root)}
#         keys = list(sub.keys())
#         keys_lower = {k: k.lower() for k in keys}
#
#         def pick_dir(candidates):
#             for k, kl in keys_lower.items():
#                 if any(tok in kl for tok in candidates):
#                     return sub[k]
#             return None
#
#         # 兼容多种命名：trainA/input/low 作为输入；trainB/label/gt/target/expert 作为高质
#         raw_dir = pick_dir(['trainA', 'input', 'low'])
#         exp_dir = pick_dir(['trainB', 'label', 'gt', 'target', 'expert', 'expertc', 'srgb'])
#         assert raw_dir is not None and exp_dir is not None, \
#             f"评测根目录需包含原始与高质两个子目录（支持: trainA/input/low 与 trainB/label/gt/target/expert/...）。实际: {keys}"
#
#         print(f"[PairedEvalDataset] raw_dir={raw_dir} | exp_dir={exp_dir}")
#
#         raw_list = [str(p) for p in listdir(raw_dir)]
#         exp_map = {Path(p).stem: str(p) for p in listdir(exp_dir)}
#
#         pairs = []
#         for rp in raw_list:
#             k = Path(rp).stem
#             if k in exp_map:
#                 pairs.append((rp, exp_map[k]))
#         assert len(pairs), "未在评测集中匹配到任何 (trainA, trainB/label) 配对，请检查文件名是否一致"
#
#         self.pairs = pairs
#         self.img_size = img_size
#         self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#
#     def __len__(self):
#         return len(self.pairs)
#
#     def _prep(self, pil):
#         # 如需在原始分辨率评测，可以去掉 Resize 并在外部保证两边同尺寸
#         pil = transforms.Resize([self.img_size, self.img_size])(pil)
#         ten = TF.to_tensor(pil)
#         return self.normalize(ten)
#
#     def __getitem__(self, idx):
#         raw_path, exp_path = self.pairs[idx]
#         img_raw = self._prep(Image.open(raw_path).convert('RGB'))
#         img_exp = self._prep(Image.open(exp_path).convert('RGB'))
#         name = Path(raw_path).stem
#         # 维持 (trainB, trainA, name) 的返回顺序以兼容现有 InputFetcher
#         return img_exp, img_raw, name
#
#
# # ------------------------------
# # 3) DataLoader 构造函数
# # ------------------------------
#
# def get_train_loader(root, img_size=512, resize_size=256, batch_size=8,
#                      shuffle=True, num_workers=4, drop_last=True):
#     print("\n[TrainLoader] 非成对训练集")
#     print(f"root={root} | img_size={img_size} | crop={resize_size} | bs={batch_size}")
#     dataset = UnpairedTrainDataset(root, img_size=img_size, crop_size=resize_size)
#     print(f"samples(unpaired)={len(dataset)}")
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            shuffle=shuffle,
#                            num_workers=num_workers,
#                            pin_memory=True,
#                            drop_last=drop_last)
#
#
# def get_test_loader(root, img_size=512, batch_size=8, shuffle=False, num_workers=4):
#     print("\n[TestLoader] 成对评测集（按文件名配对）")
#     dataset = PairedEvalDataset(root, img_size=img_size)
#     print(f"pairs={len(dataset)}")
#     return data.DataLoader(dataset=dataset,
#                            batch_size=batch_size,
#                            shuffle=shuffle,
#                            num_workers=num_workers,
#                            pin_memory=True)
#
#
# # ------------------------------
# # 4) InputFetcher（保留不变）
# # ------------------------------
# class InputFetcher:
#     def __init__(self, loader):
#         print(f"初始化 InputFetcher，数据加载器大小: {len(loader)}")
#         self.loader = loader
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # self.iter = None
#
#     def _fetch_refs(self):
#         try:
#             x, y, name = next(self.iter)
#         except (AttributeError, StopIteration):
#             self.iter = iter(self.loader)
#             x, y, name = next(self.iter)
#         return x, y, name
#
#     def __next__(self):
#         x, y, img_name = self._fetch_refs()
#         x, y = x.to(self.device), y.to(self.device)
#         # 约定：x=trainB（高质域样本），y=trainA（低质输入）
#         inputs = Munch(img_exp=x, img_raw=y, img_name=img_name)
#         return inputs