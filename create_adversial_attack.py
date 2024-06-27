import torchattacks
from torchattacks import LGV, BIM, MIFGSM, DIFGSM, TIFGSM,RFGSM,TPGD,PGD
import torchvision
from torchvision.models import resnet101,ResNet101_Weights
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
import os, sys
ci_build_and_not_headless = False
try:
  from cv2.version import ci_build, headless
  ci_and_not_headless = ci_build and not headless
except:
  pass
if sys.platform.startswith("linux") and ci_and_not_headless:
  os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
  os.environ.pop("QT_QPA_FONTDIR")


#img = cv.imread('wiki.jpg', cv.IMREAD_GRAYSCALE)


def image_preprocessing(image_dir:str,device)->list:
  image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
  tensor = torch.tensor(image, dtype=torch.float32,requires_grad=True,device=device)
  H,W,C=tensor.shape
  tensor=tensor.reshape(C,H,W)
  tensor = tensor.unsqueeze(0)
  return image,tensor, H,W,C


  
def create_double_attacks(model:torchvision.models)->list:
    atk1 = torchattacks.FGSM(model, eps=8/255)
    atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=40, random_start=True)
    return atk1,atk2


def histogram(image:np.ndarray):
  img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  assert img is not None, "file could not be read, check with os.path.exists()"
  hist,bins = np.histogram(img.flatten(),256,[0,256])
  cdf = hist.cumsum()
  cdf_normalized = cdf * float(hist.max()) / cdf.max()
  plt.plot(cdf_normalized, color = 'b')
  plt.hist(img.flatten(),256,[0,256], color = 'r')
  plt.xlim([0,256])
  plt.legend(('cdf','histogram'), loc = 'upper left')
  plt.show()
  
  
def launch_attacks(model:torchvision.models,image:torch.Tensor,label:torch.Tensor)->torch.Tensor:
  atk1,atk2 = create_double_attacks(model=model)
  attack = torchattacks.MultiAttack([atk1, atk2])
  adv_images = attack(image, label)
  return adv_images

def postprocessing(adv_image:torch.Tensor,H,W,C:int)->np.ndarray:
  adv_image = torch.squeeze(adv_image)
  adv_image = adv_image.detach().cpu().numpy().reshape(H, W, C)
  adv_image = cv2.cvtColor(adv_image, cv2.COLOR_BGR2RGB)
  return adv_image
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet101(pretrained=True,weights=ResNet101_Weights.DEFAULT).to(device).eval()
atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print(atk)
image,tensor, H,W,C=image_preprocessing(device=device,image_dir="/home/tarek/projects/cameras-simulation-tool/src/uuv_simulator/uuv_gazebo_worlds/Media/materials/textures/Rusty-mat.jpg")
labels=torch.tensor(data=[1],device=device)
adv_images = atk(tensor, labels)
print(adv_images.shape)
adv_image=postprocessing(adv_image=adv_images,H=H,W=W,C=C)
histogram(image=image)
histogram(image=adv_image)
cv2.imwrite("test.png",adv_image)
