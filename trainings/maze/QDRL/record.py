# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


import sys
import json
import torch
from PIL import Image
from therlib.tools.tools import encode_img
from therutils.model.pendulum.deterministic_policy_gradient import LowDPolicyNetwork as PolicyNetwork
from therutils.model.pendulum.deterministic_policy_gradient import LowDActionValueNetworkTD3 as ActionValueNetwork
from therenv.se_star.intrusion.utils import make_env as make_intrusion_env


def record_architecture(args):
    """Construct an image of the neural network architecture.

    Args:
        args (dict): Relevant options to construct the image.
    Returns:
        dict: Visualize result of the defined neural network.
    """

    # We create the environment
    env = make_intrusion_env(args)

    # We create the actor-critic network
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    actor_net = PolicyNetwork(state_dim, action_dim, args['--hidden'])
    critic_net = ActionValueNetwork(state_dim, action_dim, args['--hidden'])

    # We send the architecture image
    return {
        **actor_net.visualize(state_dim),
        **critic_net.visualize(state_dim, action_dim)
    }


def record_model(args_path, weight_path):
    """Record a sequence of episodes from a neural network.
    Args:
        args (dict): Relevant options to play the episodes.
        weight_path (str): Path of the neural network parameters to load.
    Returns:
        dict: Sequence of episode images buffer.
    """
    with open(args_path, 'r') as file_args:
        args = json.load(file_args)
    args['--gui'] = True
    print(args)

    # Experiment options
    episode_count = 5
    width, height = 300, 300
    steps_per_second = 20
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # We create the environment
    env = make_intrusion_env(args)
    env.seed(args['--train_seed'])

    # We create the actor-critic network
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    actor_net = PolicyNetwork(state_dim, action_dim, args['--hidden']).to(device)

    # We load the weights
    checkpoint = torch.load(weight_path, map_location=device)
    actor_net.load_state_dict(checkpoint['actor_model_state_dict'])
    actor_net.eval()

    # We view the model
    data = {}

    with torch.no_grad():
        for episode in range(episode_count):
            done = False
            data['episode_{}'.format(episode)] = []

            state = env.reset(intruder_position=[100, 100])
            episode_reward = 0
            step = 0

            while not done:
                step += 1
                print('step', step)
                img = Image.fromarray(env.render(mode='rgb_array'))
                data['episode_{}'.format(episode)].append(encode_img(img))
                model_state = torch.FloatTensor(state).to(device)
                action = actor_net(model_state).detach().cpu().numpy()
                # action = ou_noise.get_action(action, step)
                state, reward, done, _ = env.step(action)
    env.close()
    return data


if __name__ == '__main__':
    args_path = sys.argv[1]
    weight_path = sys.argv[1]
    record_model(args_path, weight_path)
