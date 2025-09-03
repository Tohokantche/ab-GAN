import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms


class Generator(nn.Module):
    def __init__(
            self,
            n_classes: int,
            latent_dim: int,
            channels: int,
            img_size: int,
    ):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class GeneratorCNN(nn.Module):
    def __init__(self,
                 n_classes: int,
                 latent_dim: int,
                 channels: int,
                 ):
        super(GeneratorCNN, self).__init__()

        self.onehot = torch.zeros(n_classes, n_classes).scatter_(1, torch.tensor(list(range(n_classes))).view(n_classes, 1), 1)

        self.layer_x = (
            nn.Sequential(nn.ConvTranspose2d(latent_dim,128,3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()))
        self.layer_y = (
            nn.Sequential(nn.ConvTranspose2d(n_classes,128,3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(), ))
        self.layer_xy = (
            nn.Sequential(nn.ConvTranspose2d(256,128,3,2,0,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,channels,4,2, 1, bias=False),
            nn.Tanh()))

        self.reset_parameters()

    def forward(self, x, y):
        y = self.onehot[y]
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.layer_x(x)
        y = y.view(y.shape[0], y.shape[1], 1, 1)
        y = self.layer_y(y)
        xy = torch.cat([x,y], dim=1)
        xy = self.layer_xy(xy)
        return xy

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


class GeneratorC10(torch.nn.Module):
    def __init__(self,
                 z_dim : int = 100,
                 channels: int = 3):
        super().__init__()

        self.z_dim = z_dim
        # self.dense = torch.nn.Linear(128,512 * 4 * 4)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1, 1)),
            nn.Tanh())
        self.reset_parameters()

    def forward(self, z, y):
        # z=self.dense(z)
        # z=z.view(-1, 512, 4, 4)
        return self.model(z.view(-1, self.z_dim, 1, 1))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)