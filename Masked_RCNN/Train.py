from Configs import num_epochs, device, training_data_loader, valid_data_loader, num_classes, \
    optimizer_schedular_configs
from detection.engine import train_one_epoch, evaluate
from Model import get_model_instance_segmentation


def train():
    model = get_model_instance_segmentation(num_classes=num_classes)
    optimizer, lr_scheduler = optimizer_schedular_configs(model)
    model.to(device=device)
    return model, optimizer, lr_scheduler


if __name__ == '__main__':
    model, optimizer, lr_scheduler = train()
    for epoch in range(num_epochs):
        run = train_one_epoch(model, optimizer, training_data_loader, device, epoch, print_freq=100)
        lr_scheduler.step()
        valid = evaluate(model, valid_data_loader, device=device)
