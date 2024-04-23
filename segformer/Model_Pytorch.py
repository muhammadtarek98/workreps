import torch
from transformers import SegformerForSemanticSegmentation

from Configs import *


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.id2label = id2label
        self.model_name = "nvidia/mit-b1"
        self.label2id = label2id
        self.num_classes = len(id2label.keys())
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            reshape_last_stage=True)

    def forward(self, pixel_values, masks=None):
        output = self.model(pixel_values, masks)
        print(output.logits)
        return output.logits


"""
test=Model()
x=torch.rand((1,3,1080,1080))
output=test(x)
print(output)
"""
