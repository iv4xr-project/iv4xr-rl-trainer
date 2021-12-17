from therenv.se_star.intrusion.intrusion import Renderer
from therenv.se_star.intrusion.utils import make_env as make_intrusion_env
from therutils.parser.experiment import TrainExperiment as TrainExperimentParser
from therutils.parser.env.intrusion import Intruder as ParserIntruder
from therutils.parser.env.intrusion import Guard as ParserGuard
from therutils.parser.env.intrusion import FixedGuard as ParserFixedGuard
from therutils.parser.env.intrusion import Intrusion as ParserIntrusion
from therutils.parser.algorithm.qd_rl import ParserQDRLIntrusion
from therutils.parser.noise import EpsilonGreedy as ParserEpsilonGreedy
from therutils.model.pendulum.deterministic_policy_gradient import LowDPolicyNetwork as PolicyNetwork
import torch
import sys
import json

if __name__ == '__main__':
    args_path= sys.argv[1]
    weight_path= sys.argv[2]

    print('rfji', args_path, weight_path)

    with open(args_path) as json_file:
        args = json.load(json_file)
    # Parse the options
    parser = TrainExperimentParser(ParserIntrusion(ParserIntruder(), ParserGuard(), ParserFixedGuard()),
                                   ParserQDRLIntrusion(ParserEpsilonGreedy()))
    _args = parser.parse()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We create the environment
    env = make_intrusion_env(_args)

    # TESTT
    from PIL import Image
    state = env.reset()
    renderer = Renderer(env.unwrapped)

    # We create the actor-critic network
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    actor_net = PolicyNetwork(state_dim, action_dim, _args['--hidden']).to(device)

    # We load the weights
    checkpoint = torch.load(weight_path, map_location=device)
    actor_net.load_state_dict(checkpoint['actor_model_state_dict'])
    actor_net.eval()
    renderer.make_images_policy(actor_net)
