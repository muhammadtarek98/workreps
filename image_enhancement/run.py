import torch
import torchinfo
from UW_CycleGAN.GANTraining import GAN
from UW_CycleGAN.Dataset import CustomDataset
import pytorch_lightning as pl
from lightning import LightningModule
import os
from torchvision.transforms import v2
from UW_CycleGAN.Configs import transform
input_shape = (3, 128, 128)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
root_dir = "/home/cplus/projects/m.tarek_master/Image_enhancement/dataset/underwater_imagenet"
num_epochs = 100
device_type = "gpu" if torch.cuda.is_available() else "cpu"  # Use GPU if available
device = torch.device("cuda" if device_type == "gpu" else "cpu")

num_device = 1
data_set = CustomDataset(images_dir=root_dir,
                         device=device,
                         transform=transform)

batch_size = 2

lr = 1e-8

data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)
logger = pl.loggers.TensorBoardLogger(save_dir="UW_CycleGAN/logs",
                                      name="Underwater_CycleGAN")
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="total_generator_loss",
    min_delta=0,
    patience=0,
    verbose=False,
    mode="min",
)

checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    save_top_k=1,
    monitor="total_generator_loss",
    mode="min"
)

gan = GAN(input_shape=input_shape,
          train_loader=data_loader,
          lr=lr)
gan.to(device=device)
trainer = pl.Trainer(
    max_epochs=num_epochs,
    val_check_interval=len(data_loader),
    accelerator=device_type,
    devices=num_device,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    enable_progress_bar=True,
    fast_dev_run=False,
    enable_model_summary=True,

)

trainer.fit(model=gan, train_dataloaders=data_loader)
