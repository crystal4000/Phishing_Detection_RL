# Phishing URL Detection using Reinforcement Learning

This project uses Deep Q-Networks (DQN), a reinforcement learning technique, to detect phishing URLs. It extracts features from URL structures and trains a DQN agent to classify URLs as legitimate or phishing. The agent learns through interactions with a simulated environment, improving its classification accuracy over time.

## Files

- `dqn_model.py`: DQN architecture
- `dqn_agent.py`: DQN agent with experience replay
- `phishing_environment.py`: Simulated environment
- `feature_extractor.py`: URL feature extraction
- `url_parser.py`: URL component analysis
- `data_loader.py`: Dataset loading and preprocessing
- `train_rl.py`: DQN agent training loop
- `evaluate_model.py`: Model performance evaluation
- `config.py`: Project configuration
- `main.py`: Application entry point

## Setup

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/phishing-detection-rl.git
   cd phishing-detection-rl
   ```

2. Install stuff:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Put your URLs in `phishing_urls.json` and `legitimate_urls.json`

2. Train the model:
   ```
   python main.py --mode train
   ```
   Test model:
   ```
   python main.py --mode test
   ```

## Config

Check `config.py` to tweak things like:
- Dataset split
- Training episodes
- Batch size
- Learning rate
- Epsilon values
- NN architecture

## Results

You'll get accuracy, precision, recall, F1 score, and confusion matrix. Model saves as `phishing_detection_model.h5`, results in `evaluation_results.json`.

## Contributing

1. Fork it
2. Make a branch (`git checkout -b cool-new-feature`)
3. Commit your changes (`git commit -am 'Added a cool feature'`)
4. Push to the branch (`git push origin cool-new-feature`)
5. Create a Pull Request

## Acknowelegments
Professor Akbar Namin: [Deep Reinforcement Learning for Detecting
Malicious Websites](https://arxiv.org/pdf/1905.09207)

