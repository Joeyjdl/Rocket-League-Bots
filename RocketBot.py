import rlgym_sim as rlgym
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

from rlgym_ppo import Learner


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
    
    return rewards[2] * 2000
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0
  

  
def get_own_goal_pos(player: PlayerData):
   if player.team_num == BLUE_TEAM:
      return np.array(BLUE_GOAL_BACK)
   else:
      return np.array(ORANGE_GOAL_BACK)
  
last_reward = 0
class BalancedReward(RewardFunction):
  def reset(Self, initial_state: GameState):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:   
    rewards = np.empty(6)

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

    rewards[3] = self.calculate_defensive_reward(player, state)

    # Reward for good positioning
    # Placeholder: actual implementation depends on available state information
    rewards[4] = self.calculate_positioning_reward(player, state)

    # Penalty for negative behaviors (like own goals)
    # Placeholder: actual implementation depends on available state information
    rewards[5] = self.calculate_penalty(player, state)

    # Combine rewards (and penalties) to form a total reward
    total_reward = np.sum(rewards)
    global last_reward 
    last_reward = total_reward
    # print("Reward: " + str(total_reward) + " last: " + str(last_reward))
    return total_reward * 1000
  

  DEFENSIVE_RADIUS = 1250
  def calculate_defensive_reward(self, player: PlayerData, state: GameState) -> float:
    ball_distance_to_own_goal = np.linalg.norm(state.ball.position - get_own_goal_pos(player))
    player_distance_to_own_goal = np.linalg.norm(player.car_data.position - get_own_goal_pos(player))
    if ball_distance_to_own_goal < self.DEFENSIVE_RADIUS:
        return np.exp(-0.5 * player_distance_to_own_goal / CAR_MAX_SPEED) / 4
    return 0
  
  MIN_POSITIONING_DISTANCE = 500
  MAX_POSITIONING_DISTANCE = 2000
  def calculate_positioning_reward(self, player: PlayerData, state: GameState) -> float:
    distance_to_ball = np.linalg.norm(player.car_data.position - state.ball.position)
    if self.MIN_POSITIONING_DISTANCE < distance_to_ball < self.MAX_POSITIONING_DISTANCE:
        return 1  # Simple binary reward for being in a good position
    return 0
  
  OWN_GOAL_VELOCITY_THRESHOLD = 750  
  def calculate_penalty(self, player: PlayerData, state: GameState) -> float:
    # ball_velocity_towards_own_goal = np.dot(state.ball.velocity, player.own_goal_direction)
    # if ball_velocity_towards_own_goal > self.OWN_GOAL_VELOCITY_THRESHOLD:
    #     return -1  # Penalty for moving the ball towards own goal
    return 0
  

print(f"Code has begun")
#Make the default rlgym environment
env = rlgym.make(reward_fn=BalancedReward(), copy_gamestate_every_step=True)

def init_rlgym():
  import rlgym_sim
  return rlgym_sim.make(reward_fn=BalancedReward())


class CustomCallback(BaseCallback):
  global last_reward
  def __init__(self, verbose=0):
    super().__init__(verbose)
  def _on_step(self) -> bool:
     self.logger.record("reward/total_reward", last_reward)
     return True


# Train the model
print(f"Training has begun")
for i in range(5):
    save_path = "./BotBrains/RockBotBrainBalanced" + str(i)
    model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./logs")
    tempModel = model.learn(total_timesteps=int(5e6), tb_log_name="Run_" + str(i) , progress_bar=True, callback=CustomCallback())
    PPO.save(tempModel, path=save_path)

