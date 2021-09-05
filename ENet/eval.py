import numpy as np
import torch
from utils import get_cfs_matrix, calculate_mean_iou
from tqdm import tqdm


def eval(dataloader, model, loss_fn, device="cpu", num_classes=2):
    """
    model has been sent before this function is called
    """
    model.eval()
    test_losses = []
    cfs_matrix = np.zeros((num_classes, num_classes))
    loader = tqdm(dataloader)
    loader.set_description("Evaluating...")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)  # batchsize , num_classes , img_height, img_width
            loss = loss_fn(pred, target)
            test_losses.append(loss.item())
            cfs_matrix += get_cfs_matrix(pred.detach(), target.detach())

    mean_iou = calculate_mean_iou(cfs_matrix)
    model.train()
    return sum(test_losses) / len(test_losses), mean_iou