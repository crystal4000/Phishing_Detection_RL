import json
import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from data_loader import load_dataset
from dqn_agent import DQNAgent
from phishing_environment import PhishingEnvironment
from train_rl import train_dqn
from test_rl import test_model
from evaluate_model import evaluate_model
from convert_numpy_to_python import convert_numpy_to_python
from config import *

def main():
    parser = argparse.ArgumentParser(description='Train or test the phishing detection model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode to run the script in: train or test (default: train)')
    args = parser.parse_args()

    try:
        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        print(f"Using device: {DEVICE}")

        print("Loading dataset...")
        dataset = load_dataset(PHISHING_URLS_FILE, LEGITIMATE_URLS_FILE, TEST_SPLIT)

        train_features = dataset['train_features']
        train_labels = dataset['train_labels']
        test_features = dataset['test_features']
        test_labels = dataset['test_labels']

        print(f"Dataset loaded. Training samples: {len(train_features)}, Test samples: {len(test_features)}")

        if args.mode == 'train':

            print(f"Dataset loaded. Training samples: {len(train_features)}, Test samples: {len(test_features)}")
            print(f"Shape of train_features: {np.array(train_features).shape}")
            print(f"Shape of train_labels: {np.array(train_labels).shape}")
            print(f"Type of train_features: {type(train_features)}")
            print(f"Type of train_labels: {type(train_labels)}")

            # Create the environment
            print("Creating environment...")
            env = PhishingEnvironment(train_features, train_labels)
            print(f"Environment created. Number of samples: {env.num_samples}")

            # Create the agent
            print("Creating agent...")
            agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
            print("Agent created.")

            # Check if there's a final trained model
            if os.path.exists('final_model.pth'):
                print("Found a final trained model: final_model.pth")
                user_input = input("Do you want to use the final trained model? (y/n): ")
                if user_input.lower() == 'y':
                    agent.model.load_state_dict(torch.load('final_model.pth'))
                    agent.target_model.load_state_dict(torch.load('final_model.pth'))
                    print("Final model loaded. Skipping training and proceeding to evaluation.")
                    trained_agent = agent
                else:
                    print("Proceeding with training...")
                    # Check for checkpoints
                    checkpoints = [f for f in os.listdir('.') if f.startswith('model_checkpoint_episode_') and f.endswith('.pth')]
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        print(f"Found checkpoint: {latest_checkpoint}")
                        user_input = input("Do you want to load the latest checkpoint? (y/n): ")
                        if user_input.lower() == 'y':
                            checkpoint = torch.load(latest_checkpoint)
                            agent.model.load_state_dict(checkpoint)
                            agent.target_model.load_state_dict(checkpoint)
                            start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
                            print(f"Checkpoint loaded. Resuming from episode {start_episode}")
                        else:
                            start_episode = 0
                            print("Starting training from the beginning.")
                    else:
                        start_episode = 0
                        print("No checkpoints found. Starting training from the beginning.")
                    
                    print("Starting training...")
                    trained_agent = train_dqn(env, agent, EPISODES, BATCH_SIZE, save_interval=SAVE_INTERVAL, start_episode=start_episode)
            else:
                print("No final trained model found. Starting training from the beginning.")
                start_episode = 0
                print("Starting training...")
                trained_agent = train_dqn(env, agent, EPISODES, BATCH_SIZE, save_interval=SAVE_INTERVAL, start_episode=start_episode)

            trained_agent.model.eval()
            torch.save(trained_agent.model.state_dict(), 'final_model.pth')
            
            print("Training completed. Evaluating model...")
            evaluation_results = evaluate_model(trained_agent, test_features, test_labels)

            # Save the trained model
            print(f"Saving trained model to {MODEL_SAVE_PATH}")
            torch.save(trained_agent.model.state_dict(), MODEL_SAVE_PATH)

            evaluation_results = convert_numpy_to_python(evaluation_results)

            # Save evaluation results
            print(f"Saving evaluation results to {RESULTS_SAVE_PATH}")
            with open(RESULTS_SAVE_PATH, 'w') as f:
                json.dump(evaluation_results, f, indent=4)

            print("Evaluation results:")
            print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"Precision: {evaluation_results['precision']:.4f}")
            print(f"Recall: {evaluation_results['recall']:.4f}")
            print(f"F1 Score: {evaluation_results['f1']:.4f}")
            print(f"False Positive Rate: {evaluation_results['fpr']:.4f}")
            print(f"True Positive Rate: {evaluation_results['tpr']:.4f}")
            print(f"False Negative Rate: {evaluation_results['fnr']:.4f}")

        elif args.mode == 'test':
            # Test the model (for both train and test modes)
            print("\nTesting the model on test data...")
            print(f"Shape of test_features: {np.array(test_features).shape}")
            print(f"Shape of test_labels: {np.array(test_labels).shape}")
            print(f"Type of test_features: {type(test_features)}")
            print(f"Type of test_labels: {type(test_labels)}")
            model = DQNAgent(STATE_SIZE, ACTION_SIZE)
            model.model.load_state_dict(torch.load('final_model.pth'))
            test_results = test_model(model.model, test_features, test_labels)
            test_predictions, test_accuracy = test_results[:2]

            # Calculate additional metrics
            precision = precision_score(test_labels, test_predictions)
            recall = recall_score(test_labels, test_predictions)
            f1 = f1_score(test_labels, test_predictions)
            cm = confusion_matrix(test_labels, test_predictions)

            # Save test predictions and metrics
            test_results = {
                'predictions': test_predictions,
                'true_labels': test_labels,
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            }

            TEST_RESULTS_PATH = 'test_results.json'
            print(f"Saving test results to {TEST_RESULTS_PATH}")
            with open(TEST_RESULTS_PATH, 'w') as f:
                json.dump(convert_numpy_to_python(test_results), f, indent=4)

            print("Program completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()