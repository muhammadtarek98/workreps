import matplotlib.pyplot as plt
import numpy as np
from Model_PL import SegformerFinetuner
import Configs
import torch
import cv2
import utils
from collections import OrderedDict
import torchinfo

new_state_dict = OrderedDict()
img_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/ROV Launch and Cook Islands Nodules on seafloor/00002314.jpg"
image = cv2.resize(cv2.imread(img_path),
                   dsize=(Configs.h,
                          Configs.w),
                   interpolation=cv2.INTER_LINEAR)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ckpt_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/logs/segformer_logs_b1_with_combined_data_distribution/version_0/checkpoints/epoch=9-step=5572.ckpt"
model_weights = torch.load(ckpt_path)
for key_name, value in model_weights["state_dict"].items():
    if "model_torch_class.model." in key_name:
        continue
    else:
        new_state_dict[key_name.replace("model.", "")] = value

model = SegformerFinetuner().to(device=Configs.device)
#print(model)
model.load_state_dict(state_dict=model_weights["state_dict"])
"""torchinfo.summary(model=model,
                  input_size=image.shape,
                  device=Configs.device)"""
feature_extractor = Configs.create_feature_extractor(mean=Configs.mean_3,
                                                     std=Configs.std_3)
encoded_inputs = feature_extractor(image, return_tensors="pt")
"""preprocessed_image = np.array(encoded_inputs["pixel_values"].squeeze_())
preprocessed_image = preprocessed_image.reshape(preprocessed_image.shape[1],
                                                preprocessed_image.shape[2],
                                                preprocessed_image.shape[0])"""
#plt.imshow(preprocessed_image)
#plt.show()
encoded_inputs.to(device=Configs.device)

model.eval()
print(encoded_inputs["pixel_values"].shape)
encoded_output = model(encoded_inputs["pixel_values"])
logits = encoded_output.logits
upsampled_logits = torch.nn.functional.interpolate(logits,
                                                   size=(Configs.h,
                                                         Configs.w),
                                                   mode=Configs.up_sampling)
print(upsampled_logits.shape)
pred_seg = upsampled_logits.argmax(dim=1)
pred_seg = pred_seg.squeeze_().detach().cpu().numpy()
#print(np.unique(pred_seg))
plt.imshow(pred_seg, cmap="binary")
plt.show()
