import os
import time
from typing import Tuple
import numpy as np
import torch
import continuous_policy


class Agent:
    def __init__(self):
        print("Bot initializing...")
        policy_layer_sizes: Tuple[int, ...] = (256, 256, 256)
        continuous_var_range: Tuple[float, ...] = (0.1, 1.0),
        self.actor = continuous_policy.ContinuousPolicy(
            89,
            16,
            policy_layer_sizes,
            device="cuda",
        ).to(torch.device("cuda"))

        print("Bot loading...")
        missing_keys = self.actor.load_state_dict(
            torch.load(os.path.join("C:/Users/nhcla/Documents/Uni/Machine Learning/Rocket-League-Bots/rlbot/src/Brain", "PPO_POLICY.pt"))
        )
        print(f"missing keys {missing_keys}")
        # self.policy.get_action(obs)
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        return

    def act(self, state):
        # Evaluate your model here
        # action = [1, 0, 0, 0, 0, 0, 0, 0]

        actions, log_probs = self.actor.get_action(obs=state)
        actions = actions.numpy().astype(np.float32)

        return actions
