import Configs
from Model_PL import SegformerFinetuner
import pytorch_lightning as pl

segformerfinetuner = SegformerFinetuner()
segformerfinetuner.to(device=Configs.device)
trainer = pl.Trainer(max_epochs=Configs.num_epochs,
                     val_check_interval=len(Configs.valid_dataloader),
                     accelerator=Configs.device_type,
                     devices=Configs.num_devices,
                     callbacks=[Configs.early_stop_callback,
                                Configs.checkpoint_callback,
                                Configs.lr_monitor],
                     logger=Configs.logger,
                     enable_progress_bar=True,
                     enable_model_summary=True)
"""
print(segformerfinetuner.model.config.num_labels)
print(segformerfinetuner.model.config.id2label)
print(segformerfinetuner.model.config.label2id)

"""
trainer.fit(model=segformerfinetuner,
            val_dataloaders=Configs.valid_dataloader,
            train_dataloaders=Configs.train_dataloader,

            )
