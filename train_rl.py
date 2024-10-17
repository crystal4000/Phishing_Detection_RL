# %%writefile train_rl.py
import torch
from tqdm import tqdm


def train_dqn(env, agent, episodes=1000, batch_size=32, save_interval=50, start_episode=0):
    print(f"Starting training with {episodes} episodes and batch size {batch_size}")
    print(f"Environment has {env.num_samples} samples")

    for episode in tqdm(range(start_episode, episodes)):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Agent chooses an action
            action = agent.act(state)

            # Take action and observe the result
            next_state, reward, done = env.step(action)
            total_reward += reward


            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, reward, state, done)  # Use current state if next_state is None

            state = next_state if next_state is not None else state

            # Perform experience replay if enough memories are stored
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Update target network periodically
        if episode % 10 == 0:
            agent.target_train()

        # Save model periodically
        if (episode + 1) % save_interval == 0:
            save_path = f'model_checkpoint_episode_{episode + 1}.pth'
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Print episode results (optional)
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent