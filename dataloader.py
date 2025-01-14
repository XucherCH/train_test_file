import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Reshape 类的实例化


# 自定义数据集类
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道掩码

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)


        return image, mask

# 数据预处理
image_transform = transforms.Compose([
    transforms.Resize((640*5, 640*5)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# 实例化训练数据集
train_image_dir = "../dataset/train_array"  # 替换为训练图像的路径
train_mask_dir = r"../dataset/train_mask_enhanced"  # 替换为训练掩码的路径

train_dataset = ImageMaskDataset(image_dir=train_image_dir, mask_dir=train_mask_dir,
                                 transform=image_transform, mask_transform=mask_transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 实例化验证数据集
val_image_dir = r"../dataset/val_array_reshape"  # 替换为验证图像的路径
val_mask_dir = r"../dataset/val_mask"  # 替换为验证掩码的路径

val_dataset = ImageMaskDataset(image_dir=val_image_dir, mask_dir=val_mask_dir,
                               transform=image_transform, mask_transform=mask_transform)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 训练数据加载器循环

