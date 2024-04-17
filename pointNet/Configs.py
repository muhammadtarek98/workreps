import torch, torch_geometric, pytorch_lightning, lightning
print(torch.__version__)
print(torch_geometric.__version__)
print(pytorch_lightning.__version__)
num_global_feature = 1024
dim = 3
labels = 2
lr = 1e-4
batch_size = 64
epochs = 1
logger = lightning.pytorch.loggers.TensorBoardLogger(save_dir="logs", name="pointnet")
early_stop_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min")
checkpoint_callback = pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint(save_top_k=1, monitor="val_loss")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
