from typing import Union, Dict, Any, Tuple, Optional

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.autograd import Variable
from torchvision.utils import make_grid
from src.utils.utils import  abLoss


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        objective_type : str,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.objective_type = objective_type
        if self.objective_type not in ["Rg_LS", "LS", "abLoss"]:
            raise ValueError(f"Objective type {self.objective_type} is not supported. Please choose from ['LS', 'abLoss'].")
        if self.objective_type == "LS":
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.objective_type == "abLoss":
            self.adversarial_loss = abLoss(1.0, 1.0)
        self.fixed_noise = Variable(torch.randn(16, self.hparams.latent_dim))
        self.fixed_label = Variable(torch.tensor(np.concatenate((
            np.random.choice(10, 10, replace=False),
            np.random.choice(10, 6, replace=False)))))


    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("test", k)): v for k, v in log_dict.items()})
        return None

    # Real_data_target and gen_data_target can be merged, we only separated them for clarity
    @staticmethod
    def real_data_target(size:int, l_value:float=1) -> Tensor:
        data = Variable(l_value*torch.ones(size, 1))
        return data

    @staticmethod
    def gen_data_target(size:int, l_value:float=-1) -> Tensor:
        data = Variable(l_value*torch.ones(size, 1))
        return data

    def gen_noise(self, size:int) -> Tensor:
        n = Variable(torch.randn(size, self.hparams.latent_dim))
        return n

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:

        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # Create adversarial ground truths
        real_data = Variable(imgs)
        real_data_target = MNISTGANModel.real_data_target(batch_size)
        gen_data_target = MNISTGANModel.gen_data_target(batch_size)

        # Create noise and labels for generator input
        noise_data = self.gen_noise(batch_size)
        gen_input_labels = Variable(labels)

        if optimizer_idx == 0 or not self.training:
            #  generate images and calculate the adversarial loss for the generator
            # When optimizer_idx == 0 the model is optimizing the generator

            #  Generate a batch of images
            gen_data = self.generator(noise_data, gen_input_labels)

            #  Calculate loss to measure generator's ability to fool the discriminator
            prediction_d = self.discriminator(gen_data, gen_input_labels)
            loss = self.adversarial_loss(prediction_d, real_data_target)* (0.5 if self.objective_type == "LS" else 1)

            log_dict["g_loss"] = loss.item()

        if optimizer_idx == 1 or not self.training:
            #  Generate images and calculate the adversarial loss for the discriminator
            # When optimizer_idx == 1 the model is optimizing the discriminator

            #  Generate a batch of images
            gen_data = self.generator(noise_data, gen_input_labels)

            #  Calculate loss for real images
            real_d_error = self.adversarial_loss(self.discriminator(real_data, gen_input_labels), real_data_target)

            #  Calculate loss for fake images
            gen_d_error = self.adversarial_loss(self.discriminator(gen_data, gen_input_labels), gen_data_target)

            #  Calculate total discriminator loss
            loss = (real_d_error + gen_d_error) * (0.5 if self.objective_type == "LS" else 1)

            log_dict["d_loss"] = loss.item()

        return log_dict, loss

    def on_epoch_end(self):

        #  Create fake images
        gen_data = self.generator(self.fixed_noise, self.fixed_label)
        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                #  Log fake images to wandb (https://docs.wandb.ai/guides/track/log/media)

                grid = make_grid(gen_data, nrow=8, padding=2, normalize=True)
                image = grid.permute(1, 2, 0).data.numpy()
                logger.experiment.log({"gen_imgs": wandb.Image(image)})
