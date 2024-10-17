# %%writefile data_loader.py
import json
import random
from url_parser import parse_url
from feature_extractor import extract_features

def load_dataset(phishing_file, legitimate_file, test_split=0.2):
    try:
        # Load phishing URLs
        with open(phishing_file, 'r') as f:
            phishing_urls = json.load(f)

        # Load legitimate URLs
        with open(legitimate_file, 'r') as f:
            legitimate_urls = json.load(f)

        print(f"Loaded {len(phishing_urls)} phishing URLs and {len(legitimate_urls)} legitimate URLs")

        # Combine and shuffle the data
        all_urls = [(url, 1) for url in phishing_urls] + [(url, 0) for url in legitimate_urls]
        random.shuffle(all_urls)

        # Split into training and testing sets
        split_index = int(len(all_urls) * (1 - test_split))
        train_data = all_urls[:split_index]
        test_data = all_urls[split_index:]

        print(f"Split data into {len(train_data)} training samples and {len(test_data)} test samples")

        # Preprocess the data
        train_features, train_labels = preprocess_data(train_data)
        test_features, test_labels = preprocess_data(test_data)

        return {
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels
        }
    except Exception as e:
        print(f"Error in load_dataset: {str(e)}")
        raise

def preprocess_data(data):
    features = []
    labels = []

    for url, label in data:
        try:
            # Ensure the URL has a scheme
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed_url = parse_url(url)
            url_features = extract_features(url)
            features.append(url_features)
            labels.append(label)
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")

    return features, labels

# # Example usage
# if __name__ == "__main__":
#     phishing_file = "data_phishing_37175.json"
#     legitimate_file = "data_legitimate_36400.json"

#     dataset = load_dataset(phishing_file, legitimate_file)

#     print(f"Number of training samples: {len(dataset['train_features'])}")
#     print(f"Number of testing samples: {len(dataset['test_features'])}")
#     if dataset['train_features']:
#         print(f"Number of features per URL: {len(dataset['train_features'][0])}")