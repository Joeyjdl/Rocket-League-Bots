import rlgym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback

import os


from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
import numpy as np



class SpeedReward(RewardFunction):
  def reset(self, initial_state: GameState):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    linear_velocity = player.car_data.linear_velocity
    
    rewards = np.empty(4)

    #reward for going fast
    rewards[0] = math.vecmag(linear_velocity) / (CAR_MAX_SPEED*2)

    #reward for getting close to the ball
    dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
    rewards[1] = np.exp(-0.5 * dist / CAR_MAX_SPEED)

    #reward for speed towards the ball
    vel = player.car_data.linear_velocity
    pos_diff = state.ball.position - player.car_data.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
    vel /= CAR_MAX_SPEED
    rewards[2] = float(np.dot(norm_pos_diff, vel))

    #reward for facing the ball
    pos_diff = state.ball.position - player.car_data.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
    rewards[3] = float(np.dot(player.car_data.forward(), norm_pos_diff))
    
    return np.sum(rewards)
    
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
env = rlgym.make(game_speed=100, reward_fn=SpeedReward())

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
