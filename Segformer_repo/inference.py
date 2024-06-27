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
img_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/115351AA.mp4_/10_left.jpg"
img_path_2="/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/ROV Launch and Cook Islands Nodules on seafloor/00005195.jpg"
img_path_3="/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/datasets/ROV Launch and Cook Islands Nodules on seafloor/00005134.jpg"
image_1 = cv2.imread(img_path)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2=cv2.cvtColor(cv2.imread(img_path_2),cv2.COLOR_BGR2RGB)
image_3=cv2.cvtColor(cv2.imread(img_path_3),cv2.COLOR_BGR2RGB)

ckpt_path = "/home/cplus/projects/m.tarek_master/gravel_2D/graval_detection_project/transformers_for_CV/Segformer_repo/dataset_v2_logs/segformer_logs_b1_without_normalization/version_0/checkpoints/epoch=199-step=115613.ckpt"
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
feature_extractor = Configs.create_feature_extractor()
encoded_inputs = feature_extractor(image_1, return_tensors="pt")
encoded_input_2= feature_extractor(image_2, return_tensors="pt")
encoded_input_3=feature_extractor(image_3, return_tensors="pt")
"""preprocessed_image = np.array(encoded_inputs["pixel_values"].squeeze_())
preprocessed_image = preprocessed_image.reshape(preprocessed_image.shape[1],
                                                preprocessed_image.shape[2],
                                                preprocessed_image.shape[0])"""
#plt.imshow(preprocessed_image)
#plt.show()
encoded_inputs.to(device=Configs.device)
encoded_input_2.to(device=Configs.device)
encoded_input_3.to(device=Configs.device)

model.eval()
print(encoded_inputs["pixel_values"].shape)
encoded_output = model(encoded_inputs["pixel_values"])
logits = encoded_output.logits
upsampled_logits = torch.nn.functional.interpolate(logits,
                                                   size=(Configs.h,
                                                         Configs.w),
                                                   mode=Configs.up_sampling,
                                                   align_corners=True)
print(upsampled_logits.shape)
pred_seg = upsampled_logits.argmax(dim=1)
pred_seg = pred_seg.squeeze_().detach().cpu().numpy()
plt.imshow(utils.prediction_to_vis(prediction=pred_seg))
plt.show()


encoded_output_2=model(encoded_input_2["pixel_values"])
logits_2=encoded_output_2.logits
upsampled_logits = torch.nn.functional.interpolate(logits_2,
                                                   size=(Configs.h,
                                                         Configs.w),
                                                   mode=Configs.up_sampling,
                                                   align_corners=True)

pred_seg = upsampled_logits.argmax(dim=1)
pred_seg = pred_seg.squeeze_().detach().cpu().numpy()
plt.imshow(utils.prediction_to_vis(prediction=pred_seg))
plt.show()


encoded_output_3=model(encoded_input_3["pixel_values"])
logits_3=encoded_output_3.logits
upsampled_logits = torch.nn.functional.interpolate(logits_3,
                                                   size=(Configs.h,
                                                         Configs.w),
                                                   mode=Configs.up_sampling,
                                                   align_corners=True)

pred_seg = upsampled_logits.argmax(dim=1)
pred_seg = pred_seg.squeeze_().detach().cpu().numpy()



#print(np.unique(pred_seg))
plt.imshow(utils.prediction_to_vis(prediction=pred_seg))





plt.show()


