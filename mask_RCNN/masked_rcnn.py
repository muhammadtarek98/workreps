from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchinfo import summary


def get_model(num_classes: int,training_flag=None) -> MaskRCNN:
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features_boxes = model.roi_heads.box_predictor.cls_score.in_features
    in_features_masks = model.roi_heads.mask_predictor.conv5_mask.in_channels
    out_features_masks = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_boxes, num_classes=2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_masks, dim_reduced=out_features_masks,
                                                       num_classes=num_classes)

    for param in model.parameters():
            param.requires_grad = True
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