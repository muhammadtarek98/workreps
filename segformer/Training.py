from Model_PL import SegformerFinetuner
from Configs import device, id2label, train_dataloader, valid_dataloader, test_dataloader, model_name, logger, \
    early_stop_callback, checkpoint_callback, device_type, num_devices, num_epochs
import pytorch_lightning as pl

segformerfinetuner = SegformerFinetuner()
segformerfinetuner.to(device=device)
trainer = pl.Trainer(max_epochs=num_epochs,
                     val_check_interval=len(valid_dataloader),
                     accelerator=device_type, devices=num_devices,
                     callbacks=[early_stop_callback, checkpoint_callback],
                     logger=logger,
                     enable_progress_bar=True,
                     fast_dev_run=True)
trainer.fit(model=segformerfinetuner,
            val_dataloaders=valid_dataloader,
            train_dataloaders=train_dataloader)
