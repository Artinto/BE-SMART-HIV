import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import imgaug
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.flip.Flipud(p=0.5),
            iaa.Sometimes(
                0.3,
                iaa.OneOf([
                    iaa.SomeOf(2, [
                        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                        iaa.ChangeColorTemperature((4000, 16000)),
                        iaa.GammaContrast((0.25, 2.0))
                    ]),
                    iaa.GaussianBlur(sigma=(0.0, 0.5))])
            ),
            iaa.Sometimes(
                0.5,
                iaa.KeepSizeByResize(iaa.Crop((170, 170), keep_size=False))
            )
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img).copy()


def data_load(dataroot):
    batch_size = 64
    aug_transforms = ImgAugTransform()

    transform = transforms.Compose([
        aug_transforms,
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = dset.ImageFolder(root=dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print("Num of dateset:", len(dataloader.dataset))

    return dataloader
