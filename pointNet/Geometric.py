import torch_geometric, torch, torchinfo
from Configs import *
from ModelPL import ModelTuner
import pytorch_lightning

geometric_shape_dataset = torch_geometric.datasets.GeometricShapes(root="data/GeometricShapes")
dataset = torch_geometric.datasets.ShapeNet(root="data/ShapeNet", include_normals=True, split="trainval",
                                            categories="Airplane")

training_spilt = int(len(dataset) * 0.8)
validation_spilt = int(len(dataset) * 0.2)
train_data = dataset[:training_spilt]
validation_data = dataset[training_spilt:]

training_loader = torch_geometric.loader.DataLoader(dataset=train_data,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    )
validation_loader = torch_geometric.loader.DataLoader(dataset=validation_data,
                                                      batch_size=16,
                                                      shuffle=False)

for i in training_loader:
    print(i)
    break

model = ModelTuner(num_points=40865,
                   labels=len(dataset),
                   num_global_feature=num_global_feature,
                   test_data=training_loader,
                   validation_data=validation_loader, dim=dim).to(device=device)

trainer = pytorch_lightning.Trainer(max_epochs=epochs,
                                    val_check_interval=len(validation_loader),
                                    accelerator="auto",
                                    devices=1,
                                    logger=logger,
                                    callbacks=[early_stop_callback, checkpoint_callback],
                                    enable_progress_bar=True,
                                    fast_dev_run=False)
trainer.fit(model=model,
            val_dataloaders=validation_loader,
            train_dataloaders=training_loader)
