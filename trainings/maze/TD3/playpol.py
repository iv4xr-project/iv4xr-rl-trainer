# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


from therutils.model.pendulum.deterministic_policy_gradient import LowDPolicyNetwork as PolicyNetwork
from therenv.se_star.intrusion.intrusion import Renderer
from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
import torch
import json
import sys

if __name__ == '__main__':
    args_path= sys.argv[1]+'/smoothdense4g2021-10-26 15:34:23.072753/exp_options.json'
    weight_path= sys.argv[1]+'/smoothdense4g2021-10-26 15:34:23.072753/agent_0/weights/weight10000.pt'


    with open(args_path) as json_file:
        args = json.load(json_file)
    # Parse the options
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We create the environment
    env = make_intrusion_env(args)

    # TESTT
    from PIL import Image
    state = env.reset()
    renderer = Renderer(env.unwrapped)

    # We create the actor-critic network
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    actor_net = PolicyNetwork(state_dim, action_dim, args['--hidden']).to(device)

    # We load the weights
    checkpoint = torch.load(weight_path, map_location=device)
    actor_net.load_state_dict(checkpoint['actor_model_state_dict'])
    actor_net.eval()
    renderer.make_images_policy(actor_net)
