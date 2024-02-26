import rlgym
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback

import sys

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv


from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BACK_NET_Y
import numpy as np

from rlgym_ppo import Learner
from Training import bot_exists, find_newest_checkpoint, CHECKPOINT_PATH, ExampleLogger
  
def get_own_goal_pos(player: PlayerData):
   if player.team_num == BLUE_TEAM:
      return np.array(BLUE_GOAL_BACK)
   else:
      return np.array(ORANGE_GOAL_BACK)
  
class ZeroReward(RewardFunction):
   def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
      return 0
   
   def reset(self, initial_state: GameState):
      pass
   
def build_rocketsim_env():
  #Make the default rlgym environment
  return rlgym.make(game_speed = 1, reward_fn=ZeroReward(), use_injector=True)

if __name__ == "__main__":
  print(f"Code has begun")
  if(len(sys.argv) < 2):
     print("please provide the name of a model to test")
     exit(-1)

  # 32 processes
  n_proc = 1

  # educated guess - could be slightly higher or lower
  min_inference_size = max(1, int(round(n_proc * 0.9)))

  learner = Learner(build_rocketsim_env,
                    n_proc=n_proc,
                    min_inference_size=min_inference_size,
                    metrics_logger=ExampleLogger(),
                    ppo_batch_size=50000,
                    ts_per_iteration=50000,
                    exp_buffer_size=150000,
                    ppo_minibatch_size=50000,
                    ppo_ent_coef=0.001,
                    ppo_epochs=1,
                    standardize_returns=True,
                    standardize_obs=False,
                    save_every_ts=100_000,
                    timestep_limit=5_000_000,
                    log_to_wandb=False,
                    checkpoints_save_folder= CHECKPOINT_PATH + "new_unnamed_bot" if (len(sys.argv) < 2) else (CHECKPOINT_PATH + sys.argv[1]),
                    add_unix_timestamp= (len(sys.argv) < 2)
                    )

  if len(sys.argv) == 2:
      # load from folder and use newest checkpoint
      if(bot_exists(sys.argv[1])):
          learner.load(CHECKPOINT_PATH + sys.argv[1] + "/" + find_newest_checkpoint(CHECKPOINT_PATH + sys.argv[1]), False)
  elif len(sys.argv) == 3:
      # load from folder and use specific checkpoint
      if(bot_exists(sys.argv[1])):
          learner.load(CHECKPOINT_PATH + sys.argv[1]  + "/" + sys.argv[2], False)

  learner.learn()