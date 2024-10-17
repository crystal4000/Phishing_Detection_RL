import torch
from tqdm import tqdm
from config import *
def test_model(model, test_features, test_labels):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for features, label in tqdm(zip(test_features, test_labels), total=len(test_labels), desc="Testing"):
            features = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == label).sum().item()
            predictions.append(predicted.item())

    accuracy = correct / total
    print(f'Accuracy of the model on the {total} test samples: {100 * accuracy:.2f}%')

    return predictions, accuracy