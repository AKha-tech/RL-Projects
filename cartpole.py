# pylint: disable=no-member,not-callable
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T  
import time
import pyglet

### Neural Network Setup ###
class DQN(nn.Module): # nn.Module is the usual pytorch nn library
    def __init__(self, img_height, img_width):
        # the input will be screenshot images of cartpole thus height and width
        super().__init__()

        # linear fully connected layer 1, takes the image input, and outputs to 24 nodes
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=24)

        # linear fully connected layer 2, takes the previous input, and outputs to 32 nodes
        self.fc2 = nn.Linear(in_features=24, out_features=32)

        # linear output layer, takes the previous inputs, and outputs to 2 nodes, left or right
        self.out = nn.Linear(in_features=32, out_features=2)


    def forward(self, t):
        """ pushes through the nn
        t: tensor (image)
        """
        t = t.flatten(start_dim=1) # flattening the image
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


### Experience Class ###
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

e = Experience(2,3,1,4)
# print(e) >>> Experience(state=2, action=3, next=1, reward=4)


### Replay Memory ###
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # holds stored experiences
        self.push_count = 0 # keeps track of how many experiences we've added to memory

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # when capacity this is a smart way to overwrite old ones first and keep moving through whilst overwriting
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    
    def sample(self, batch_size):
        """ sample experiences from memorey to train DQN 
        batch_size: number of samples
        """
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size) -> bool:
        return len(self.memory) >= batch_size


### Epsilon Greedy Class ###
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step) -> float:
        """ Mathematical formula for decaying epsilon
        """
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

### RL Agent ###
class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy # strategy should be an instance of epilon greedy
        self.num_actions = num_actions # num_actions: in cartpole should be 2: left/right
        self.device = device # cpu or gpu

    def select_action(self, state, policy_net):
        # policy_net: deep q network
        rate = self.strategy.get_exploration_rate(self.current_step) # from eplsilon greedy
        self.current_step += 1

        # exploration action
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        
        # exploitation action
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)


### Environment Manager ###
class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device # gpu/cpu
        self.env = gym.make('CartPole-v0').unwrapped # unwrapped gives access to 'behind the scenes'
        self.env.reset() # reset to get an initial observation
        self.current_screen = None # current_screen tracks the screen at any given time, when = None, we are at the start of the episode and haven't yet rendered the screen
        self.done = False # tracks if any action taken has ended an episode

    # wrapper functions
    def reset(self):
        self.env.reset()
        self.current_screen = None # renders screen back to start

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n # will return left, right

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item()) # step takes action and rewards tuple containg reward, ifdone...
        return torch.tensor([reward], device=self.device) # wraps the reward in a tensor

    def just_starting(self) -> bool:
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            # black_screen = np.zeros(self.current_screen.size())
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # use torchvision package to compose impage transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device) # add a bacth dimension

### Utility FUnctions ###
def plot(values, moving_avg_period):
    """ plots duration of each episode as well as the moving average of evey 100 epsiodes
    to solve cart and pole, the avg reward over 100 epsiodes must be >= 195 
    """
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Epsiode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(0.001)
    print("Episode", len(values), '\n', moving_avg_period, 'episode moving average:', get_moving_average(moving_avg_period, values)[-1])

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float) # transform values to pytorch tensor
    if len(values) >= period: # if there are enough values to calc a 'period' moving avg
        moving_avg = values.unfold(0, period, 1).mean(dim=1).flatten(start_dim=0) 
        # unfold: contains all slice with size = to the period passed in
        # take mean of each tensor and then flatten
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg)) # concat to a 0s tensor
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


### MAIN PROGRAM ###
# hyperparameters
batch_size = 256
gamma = 0.999 # from bellman equation discount rate
eps_start = 0.8
eps_end = 0.03
eps_decay = 0.008
target_update =  10 # how frequently we update target network weights with policy network weights
memory_size = 100000 # capacity of replay memory
lr = 0.001 # learning rate
num_episodes = 3000

# Initialisation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # initialise device
em = CartPoleEnvManager(device) # initialise env
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) # initialise epsilon
agent = Agent(strategy, em.num_actions_available(), device) # initialise agent
memory = ReplayMemory(memory_size) # initalise replay memory

# DQNs
policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict()) # set same weights an biases for target net as is for policy net
target_net.eval() # target net is in eval mode (not training mode)
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

### QValues Calculator Class ###
class QValues():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # static method: call these methods without creating an instance of the class
    @staticmethod
    def get_current(policy_net, states, actions): # params states and actions are smapled for repplay memory
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)) # returns q values given the state action pair

    @staticmethod
    def get_next(target_net, next_states): # calculating all of the next states/rewards
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool) # a tensor
        non_final_state_locations = (final_state_locations == False) # opposite of above
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

# Extract Tensors
def extract_tensors(experiences):
    """ Extracts a tensor of experiences into their own tensors
    """
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    try:
        t2 = torch.cat(batch.action)
    except TypeError:
        t2 = batch.action
        print(t1)
        print(batch.state)

    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)

# Training
episode_durations = []
for episode in range(num_episodes):
    # iterates over each episode
    em.reset()
    state = em.get_state()

    for timestep in count():
        # iterates over each timestep within each episode
        action = agent.select_action(state, policy_net) # if agent exploits env, policy net arg helps choose which action
        reward = em.take_action(action) # receives associated reward
        next_state = em.get_state() 
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size) # assign sample from replay mem to expereinces var
            states, actions, rewards, next_states = extract_tensors(experiences) # extract into own tensors

            current_q_values = QValues.get_current(policy_net, states, actions) # returns q value for any given state-action pair as predicted by policy net as a pytorch tensor
            next_q_values = QValues.get_next(target_net, next_states) # returns the next q values predicted by target net and next states(from experiences)
            target_q_values = (next_q_values * gamma) + rewards # callulate target q values using bellman equation

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) # mean sq error loss function
            optimizer.zero_grad() # sets gradients of all weights and biases to 0 in policy net
            loss.backward() # backpropagation 
            optimizer.step() # updates weights/biases

        if em.done:
            episode_durations.append(timestep) # stores how long this epsiode lasted
            plot(episode_durations, 100)
            break # starts a new new epsiodes
        
    # updates target net
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

em.close()



