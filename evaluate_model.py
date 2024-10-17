# evaluate_model.py
# %%writefile evaluate_model.py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(agent, features, labels):
    # Make predictions
    predictions = []
    for feature in tqdm(features, desc="Evaluating"):
        action = agent.act(feature, eval=True)  # Set eval=True to use greedy policy
        predictions.append(action)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Print results
    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate additional metrics
    true_negative = cm[0][0]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    true_positive = cm[1][1]

    # False Positive Rate
    fpr = false_positive / (false_positive + true_negative)

    # True Positive Rate (same as Recall)
    tpr = recall

    # False Negative Rate
    fnr = false_negative / (false_negative + true_positive)

    print("\nAdditional Metrics:")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"True Positive Rate: {tpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'fpr': float(fpr),
        'tpr': float(tpr),
        'fnr': float(fnr)
    }

# # Example usage
# if __name__ == "__main__":
#     from data_loader import load_dataset
#     from train_rl import train_dqn

#     # Load your dataset
#     dataset = load_dataset('phishing_urls.json', 'legitimate_urls.json')

#     # Train the model (you might want to load a pre-trained model instead)
#     trained_agent = train_dqn(dataset['train_features'], dataset['train_labels'], episodes=100, batch_size=32)

#     # Evaluate the model
#     evaluation_results = evaluate_model(trained_agent, dataset['test_features'], dataset['test_labels'])

#     # You can now use evaluation_results for further analysis or reporting