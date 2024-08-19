import torch
import torchinfo  # Import for model architecture visualization

# Import custom modules for dataset and GAN model
from GANTraining import GAN
from Dataset import CustomDataset
# Import libraries for data augmentation and image processing
import albumentations  # Optional, consider including if needed
import cv2
import torchvision

# Import PyTorch Lightning for training management
import pytorch_lightning as pl
from lightning import LightningModule  # Optional, for defining custom PyTorch Lightning modules
#import onnx for model exporting
import os
from torchvision.transforms import v2

# Define input image shape
input_shape = (3, 256, 256)  # Assuming RGB channels (3) and image resolution (120x120)

# Normalize image data (if needed)
mean = [0.67045197, 0.69555211, 0.70085958]
std = [0.18197385, 0.17502745, 0.20688226]

# Set data directory path (replace with your actual data path)
root_dir = "/home/muhammad/projects/blnk_assessment"

# Define data transformation pipeline
transforms = v2.Compose([
    torchvision.transforms.ToTensor(),  # Convert images to tensors
   torchvision.transforms.Resize(size=input_shape[1:]),  # Resize images
])

# Set training parameters
num_epochs = 10
device_type = "gpu" if torch.cuda.is_available() else "cpu"  # Use GPU if available
device = torch.device("cuda" if device_type == "gpu" else "cpu")

num_device = 1  # Number of GPUs to use (adjust as needed)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
# Load custom dataset
data_set = CustomDataset(images_dir=root_dir,device=device, transform=transforms)

# Set batch size
batch_size = 1

# Set learning rate
lr = 1e-4

# Create data loader for training
data_loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,shuffle=True, num_workers=4)
print(data_loader)
# Set up logging for training progress visualization
logger = pl.loggers.TensorBoardLogger(save_dir="logs",
                                      name="SRGAN")  # Consider using a custom logging class for more control

# Early stopping callback (optional)
early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_generator_loss",
    min_delta=0,
    patience=0,
    verbose=False,
    mode="min",
)

# Model checkpoint callback (optional)
checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    save_top_k=1,
    monitor="val_generator_loss",
)

# Set device (GPU or CPU)
device = torch.device("cpu")

# Initialize GAN model
gan = GAN(input_shape=input_shape, train_loader=data_loader, val_loader=data_loader, lr=lr)

# Move model to the chosen device (GPU or CPU)
# Create PyTorch Lightning trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    val_check_interval=len(data_loader),  # Validation after every epoch
    accelerator=device_type,
    devices=num_device,
    callbacks=[early_stop_callback, checkpoint_callback],  # Add callbacks (optional)
    logger=logger,
    enable_progress_bar=True,
    fast_dev_run=False,  # Set to True for short test runs during development (optional)
)

# Train the GAN model
trainer.fit(model=gan, train_dataloaders=data_loader, val_dataloaders=data_loader)
"""input_sample:torch.Tensor = torch.randn(size=(1, 3, 120, 120))
filepath: str = "model.onnx"
gan.eval()
gan.convert_to_onnx()
gan.convert_to_tensorrt()
generator = gan.generator
discriminator = gan.discriminator
feature_extractor = gan.feature_extractor
scripted_generator = torch.jit.script(generator)
scripted_discriminator = torch.jit.script(discriminator)
scripted_feature_extractor = torch.jit.script(feature_extractor)
torch.jit.save(scripted_generator, "scripted_generator.pt")
torch.jit.save(scripted_discriminator, "scripted_discriminator.pt")
torch.jit.save(scripted_feature_extractor, "scripted_feature_extractor.pt")

script = gan.to_torchscript(method="script",
                            example_inputs=input_sample)
torch.jit.save(script, f="model.ts")
gan.to_onnx(file_path=filepath,
            export_params=True,
            input_sample=input_sample)
"""