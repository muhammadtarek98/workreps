import torch
from transformers import SegformerForSemanticSegmentation


class Model(torch.nn.Module):
    def __init__(self, id2label, model_name, label2id, num_classes):
        super(Model, self).__init__()
        self.id2label = id2label
        self.model_name = model_name
        self.label2id = label2id
        self.num_classes = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            reshape_last_stage=True)
        self.model.config.num_labels = self.num_classes
        for para in self.model.parameters():
            para.requires_grad = True

    def forward(self, idx, mask):
        output = self.model(idx, mask)
        return output
