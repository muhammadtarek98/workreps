from torchvision.models.detection import maskrcnn_resnet50_fpn
#from torchvision.models.detection impo
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchinfo import summary
import torchvision

def get_model_instance_segmentation(num_classes:int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT",pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

"""
model = get_model(num_classes=2,training_flag=False)
t=torch.randn(size=(1,3,224,224))
model.eval()
test=model(t)
print(test.shape)
#print(type(model))
summary(model=model)
"""