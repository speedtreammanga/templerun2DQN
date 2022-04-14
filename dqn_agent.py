import torch as T
import numpy as np
# from deep_q_network import DeepQNetwork
from dqn import DeepQNetwork

# implement: https://www.youtube.com/watch?v=wc-FxNENg9U
# github: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DQN
# other simpler implementation: https://github.com/mswang12/minDQN/blob/main/minDQN.py


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=1000, eps_end=0.01, eps_dec=5e-4, device=None):
        # print(gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size, eps_end, eps_dec)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0

        self.Q_eval = DeepQNetwork(lr=self.lr, n_actions=n_actions, input_dims=input_dims, device=device)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def save(self, nb_games, avg_score, training_time):
        filename = f'nbGames{nb_games}_avgScore{np.round(avg_score, 2)}_trainingTime{training_time}_' \
                   f'batchSize{self.batch_size}_maxMemSize{self.mem_size}.pt'
        T.save(self.Q_eval.state_dict(), f'C:\\Users\\Alexandre\\Documents\\(C) Code\\ETS\\ELE767\\linear regression\\atari2\\models\\{filename}')

    def load(self, filename):
        self.Q_eval.load_state_dict(T.load(filename))
        self.Q_eval.eval()

    def store_transition(self, observation, action, reward, observation_, done):
        """
        Stores transitions (observation, action, reward, observation_, done) into
        limited memory space.
        :param observation: set of frames.
        :param action: the action chosen by the agent for the given observation.
        :param reward: the reward attributed to the given action.
        :param observation_: the next observation.
        :param done: whether or not the given action led to the end of the game.
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = observation
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.new_state_memory[index] = observation_
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def prune_transitions(self, this_many):
        index = self.mem_counter % self.mem_size - this_many

        states, actions, rewards, states_, dones = self.sample_memory(this_many-1)

        # if self.mem_counter < self.batch_size:
        #     return

        for i, (s, a, r, s_, d) in enumerate(zip(states, actions, rewards, states_, dones)):
            self.state_memory[self.mem_counter+1+i] = s.cpu()
            self.new_state_memory[self.mem_counter+1+i] = s_.cpu()
            self.reward_memory[self.mem_counter+1+i] = r.cpu()
            self.terminal_memory[self.mem_counter+1+i] = d.cpu()
            self.action_memory[self.mem_counter+1+i] = a

    def choose_action(self, observation):
        """
        Returns the most appropriate action at this time.
        At first, a random action is returned but over time, as epsilon decreases,
        the most appropriate action will be returned.
        :param observation: the current set of frames representing game state.
        :return: an action from the action space.
        """
        if np.random.rand() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(T.softmax(actions, dim=1, dtype=T.float32)).item()  # TODO: argmax being applied on linear output instead of softmax ?
        else:
            action = np.random.choice(self.action_space)

        return action

    def sample_memory(self, this_many):
        """
        Pick random samples from stored transitions.
        :return: states, actions, rewards, states_, dones
        """
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, this_many, replace=False)

        states = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        actions = self.action_memory[batch]
        rewards = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        states_ = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        dones = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        """
        Trains the Q_eval network on stored transitions.
        """
        if self.mem_counter < self.batch_size:
            return

        # batch_size = self.batch_size if self.mem_counter > self.batch_size else self.mem_counter

        self.Q_eval.optimizer.zero_grad()  # pytorch specific

        states, actions, rewards, states_, dones = self.sample_memory(this_many=self.batch_size)
        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        q_eval = self.Q_eval.forward(states)[batch_indices, actions]
        q_next = self.Q_eval.forward(states_)  # target network :around here:
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        # loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

        # if self.Q_eval.device != 'cpu':
        #     T.cuda.empty_cache()
