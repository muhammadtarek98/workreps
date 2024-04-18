"""
    test both of validation and test sets
"""
from Training import trainer, segformerfinetuner
from Configs import test_dataloader, valid_dataloader

ckpt_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/logs/segformer_logs_b1/version_1/checkpoints/epoch=91-step=52640.ckpt"

validation_results = trainer.validate(model=segformerfinetuner,
                                      ckpt_path=ckpt_path,
                                      dataloaders=valid_dataloader)

test_results = trainer.test(model=segformerfinetuner,
                            ckpt_path=ckpt_path,
                            dataloaders=test_dataloader)
