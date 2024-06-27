"""
    test both of validation and test sets
"""
from Training import trainer, segformerfinetuner
import Configs

ckpt_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/dataset_v2_logs/segformer_logs_b1_without_normalization/version_0/checkpoints/epoch=199-step=115613.ckpt"
"""training_result=trainer.validate(model=segformerfinetuner,
                                 ckpt_path=ckpt_path,
                                 dataloaders=Configs.train_dataloader)
validation_results = trainer.validate(model=segformerfinetuner,
                                      ckpt_path=ckpt_path,
                                      dataloaders=Configs.valid_dataloader)
"""
test_results = trainer.test(model=segformerfinetuner,
                            ckpt_path=ckpt_path,
                            dataloaders=Configs.inference_dataloader)
