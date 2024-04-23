"""
    test both of validation and test sets
"""
from Training import trainer, segformerfinetuner
import Configs

ckpt_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/logs/segformer_logs_b1_with_combined_data_distribution/version_0/checkpoints/epoch=9-step=5572.ckpt"
training_result=trainer.validate(model=segformerfinetuner,
                                 ckpt_path=ckpt_path,
                                 dataloaders=Configs.train_dataloader)
validation_results = trainer.validate(model=segformerfinetuner,
                                      ckpt_path=ckpt_path,
                                      dataloaders=Configs.valid_dataloader)

test_results = trainer.test(model=segformerfinetuner,
                            ckpt_path=ckpt_path,
                            dataloaders=Configs.test_dataloader)
