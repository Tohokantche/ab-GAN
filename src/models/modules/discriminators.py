import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm
from torch.nn.utils import remove_spectral_norm
from src.utils.utils import check_spectral_norm

class Discriminator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            channels: int,
            img_size: int,
    ):
        super(Discriminator, self).__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class DiscriminatorCNN(nn.Module):
    def __init__(self,
                 n_classes: int,
                 channels: int,
                 img_size: int,
                 sn: bool,
                 ):
        super(DiscriminatorCNN, self).__init__()

        self.img_size = img_size
        self.fill = torch.zeros([n_classes, n_classes, self.img_size, self.img_size])
        for i in range(n_classes):
            self.fill[i, i, :, :] = 1

        self.layer_x = (
            nn.Sequential(nn.Conv2d(channels,32,4,2, 1, bias=False),
            nn.LeakyReLU(0.2, True)))
        self.layer_y = (
            nn.Sequential(nn.Conv2d(n_classes,32, 4,2, 1, bias=False),
            nn.LeakyReLU(0.2, True)))
        self.layer_xy = nn.Sequential(
            nn.Conv2d(64, 128,4,2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3,2,0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,1,3, 1, 0, bias=False)
        )

        if sn:
            self.remove_batchnorm(self.layer_xy)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
                elif isinstance(m, nn.Linear):
                    spectral_norm(m)
                elif isinstance(m, nn.Embedding):
                    spectral_norm(m)
        self.reset_parameters()

    def forward(self, x, y):
        y= self.fill[y]
        x = self.layer_x(x)
        y = self.layer_y(y)
        xy = torch.cat([x,y], dim=1)
        xy = self.layer_xy(xy)
        xy = xy.view(xy.shape[0], -1)
        return xy

    def remove_batchnorm(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, nn.Identity())
            else:
                self.remove_batchnorm(module)

    def reset_parameters(self, root=None):
        if root is None:
            root = self
        for m in root.modules():
            is_sn = check_spectral_norm(m)
            if is_sn:
                remove_spectral_norm(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if is_sn:
                spectral_norm(m)


class DiscriminatorC10(nn.Module):
    def __init__(self,
                 channels: int=3,
                 leak :int = 0.1,
                 w_g :int = 4,
                 sn:bool=True):
        super().__init__()
        self.leak = leak
        self.w_g = w_g

        self.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1))
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1))
        self.fc = nn.Linear(w_g * w_g * 512, 1)

        if sn:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    spectral_norm(m)
                elif isinstance(m, nn.Linear):
                    spectral_norm(m)
                elif isinstance(m, nn.Embedding):
                    spectral_norm(m)
        self.reset_parameters()

    def forward(self, x, y):
        m = x
        m = nn.LeakyReLU(self.leak)(self.conv1(m))
        m = nn.LeakyReLU(self.leak)(self.conv2(m))
        m = nn.LeakyReLU(self.leak)(self.conv3(m))
        m = nn.LeakyReLU(self.leak)(self.conv4(m))
        m = nn.LeakyReLU(self.leak)(self.conv5(m))
        m = nn.LeakyReLU(self.leak)(self.conv6(m))
        m = nn.LeakyReLU(self.leak)(self.conv7(m))
        k = self.fc(m.view(-1,self.w_g * self.w_g * 512))
        return k

    def reset_parameters(self, root=None):
        if root is None:
            root = self
        for m in root.modules():
            is_sn = check_spectral_norm(m)
            if is_sn:
                remove_spectral_norm(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if is_sn:
                spectral_norm(m)