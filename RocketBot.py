import rlgym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback

import os


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv


from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y
import numpy as np



class SpeedReward(RewardFunction):
  def reset(self, initial_state: GameState):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:    
    rewards = np.empty(3)

    #reward for getting close to the ball
    dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
    rewards[0] = np.exp(-0.5 * dist / CAR_MAX_SPEED)/4

    #reward for speed towards the ball
    vel = player.car_data.linear_velocity
    pos_diff = state.ball.position - player.car_data.position
    norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
    vel /= CAR_MAX_SPEED
    rewards[1] = float(np.dot(norm_pos_diff, vel))

    #reward for getting ball to goal
    if player.team_num == BLUE_TEAM:
        objective = np.array(ORANGE_GOAL_BACK)
    else:
        objective = np.array(BLUE_GOAL_BACK)

    # Compensate for moving objective to back of net
    dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
    rewards[2] = (2 * np.exp(-0.5 * dist / BALL_MAX_SPEED))

    # print(rewards)
    
    return np.sum(rewards)
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0
  

print(f"Code has begun")
#Make the default rlgym environment
env = rlgym.make(game_speed=100, reward_fn=SpeedReward(), auto_minimize=False, use_injector=True)
# env = Monitor(env, "./logs")
# env = DummyVecEnv([lambda: env])
# # Use VecNormalize for normalization
# env = VecNormalize(env)




# Train the model
print(f"Training has begun")
for i in range(5):
    save_path = "./BotBrains/RockBotBrainNew" + str(i)
    model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./logs")
    tempModel = model.learn(total_timesteps=int(5e6), tb_log_name="Run_" + str(i) , progress_bar=True)
    PPO.save(tempModel, path=save_path)

