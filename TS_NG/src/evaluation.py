from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch
from tqdm import tqdm

from src.tsdatasets import TimeseriesDataset


def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    评估模型在给定数据加载器上的性能指标。

    Args:
        model (nn.Module): 要评估的模型。
        dataloader (torch.utils.data.DataLoader): 测试数据加载器。
        criterion (nn.Module): 损失函数。

    Returns:
        tuple: 平均损失、准确率、召回率和F1分数。
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader,"Testing..."):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            # Using argmax to get the most probable class
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, recall, f1
