# %%writefile phishing_environment.py
import numpy as np

class PhishingEnvironment:
    def __init__(self, features, labels):
        print(f"Initializing PhishingEnvironment with features type: {type(features)} and labels type: {type(labels)}")
        self.features = features
        self.labels = labels
        self.current_index = 0
        print(f"Features length: {len(self.features)}")
        self.num_samples = len(self.features)

    def reset(self):
        self.current_index = 0
        return self.features[self.current_index]

    def step(self, action):
        # Get the true label for the current URL
        true_label = self.labels[self.current_index]

        # Compute reward
        if action == true_label:
            reward = 1  # Correct classification
        else:
            reward = -1  # Incorrect classification

        # Move to the next sample
        self.current_index += 1

        # Check if we've reached the end of the dataset
        if self.current_index >= self.num_samples:
            done = True
            next_state = None
        else:
            done = False
            next_state = self.features[self.current_index]

        return next_state, reward, done

    def render(self):
        print(f"Current Index: {self.current_index}")
        print(f"Current Feature: {self.features[self.current_index]}")
        print(f"True Label: {self.labels[self.current_index]}")

    def get_state(self):
        return self.features[self.current_index]

    def get_label(self):
        return self.labels[self.current_index]