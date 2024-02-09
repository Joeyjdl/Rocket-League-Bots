import rlgym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback

import os


from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np


class SpeedReward(RewardFunction):
  def reset(self, initial_state: GameState):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    linear_velocity = player.car_data.linear_velocity
    reward = math.vecmag(linear_velocity)
    
    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0
  

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, model):
        super(SaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.model = model

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            PPO.save(self.model, path=self.save_path)
        return True


print(f"Code has begun")
#Make the default rlgym environment
env = rlgym.make(reward_fn=SpeedReward())

#Load model or initialize PPO from stable_baselines3
print(f"Loading has begun")
if os.path.exists('./BotBrains/RockBotBrain.zip'):
    model = PPO.load('./BotBrains/RockBotBrain', env=env, verbose=1)
else:
    model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
# Set up the callback to save the model every `save_freq` steps
save_freq = 100  # Adjust as needed
save_path = "./BotBrains/RockBotBrain"
save_callback = SaveModelCallback(save_freq, save_path, model)

# Train the model with the callback
print(f"Training has begun")
model.learn(total_timesteps=int(1e6), callback=save_callback)
