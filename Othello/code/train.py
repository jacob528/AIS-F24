import torch
import torch.optim as optim
import numpy as np
from model import ReversiNet
from replay_buffer import ReplayBuffer
from game import ReversiGame

def train_model():
    input_size = 64
    output_size = 64
    model = ReversiNet(input_size, output_size)
    target_model = ReversiNet(input_size, output_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    buffer = ReplayBuffer(10000)
    loss_fn = torch.nn.MSELoss()

    game = ReversiGame()
    num_episodes = 1000
    batch_size = 64
    target_update_freq = 50

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    for episode in range(num_episodes):
        state = game.reset()
        done = False
        loss = None

        while not done:
            state_tensor = torch.tensor((state + 1) / 2, dtype=torch.float32).flatten().unsqueeze(0)
            if np.random.rand() < epsilon:
                action = game.get_random_valid_move()
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            # 检查 action 是否为 None
            if action is None or not game.is_valid_move(action):
                action = game.get_random_valid_move()
                # 如果仍然没有合法动作，跳过本回合
                if action is None:
                    game.current_player *= -1
                    continue

            next_state, reward, done = game.step(action)
            buffer.push(state, action, reward / 100.0, next_state)

            state = next_state

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                states, actions, rewards, next_states = zip(*batch)

                states = torch.tensor((np.array(states) + 1) / 2, dtype=torch.float32).view(batch_size, -1)
                next_states = torch.tensor((np.array(next_states) + 1) / 2, dtype=torch.float32).view(batch_size, -1)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                q_pred = model(states)
                q_next = target_model(next_states).detach()

                q_target = q_pred.clone()
                for i in range(batch_size):
                    q_target[i][actions[i]] = rewards[i] + 0.99 * q_next[i].max()

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if loss is not None and episode % 100 == 0:
            print(f"Episode {episode}, Loss (log scale): {torch.log(loss).item()}")

    torch.save(model.state_dict(), "reversi_model.pth")
    print("Model trained and saved.")
