# import Training as functions
from utils.logger.ExampleLogger import ExampleLogger
from utils.envBuilder.build_rocketsim_env import build_rocketsim_env
from utils.stateSetter.stateSetter import CustomState

import time
import sys
import os
import rlgym_sim  
import numpy as np
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
    EventReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils import common_values
from rlgym_sim.utils.state_setters import RandomState


CHECKPOINT_PATH = "data/checkpoints/"
LOG_TO_WANDB = False

def bot_exists(name):
    folder = os.path.abspath("./data/checkpoints/" + name)
    return os.path.exists(folder)

def find_newest_checkpoint(path):
    abs_path = os.path.abspath(path)

    max_number = -1
    max_folder_name = None

    if not os.path.isdir(abs_path):
        print("Invalid directory path.")
        return None

    # Iterate over directories in the given path
    for folder_name in os.listdir(abs_path):
        if folder_name.isdigit():
            number = int(folder_name)
            if number > max_number:
                max_number = number
                max_folder_name = folder_name

    return max_folder_name


# ## Load model
if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    if len(sys.argv) < 2:
        print("Please enter a name for the model you would like to create/load")
        exit(-1)

    # 32 processes
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=50_000_000_000,
                      log_to_wandb=LOG_TO_WANDB,
                      checkpoints_save_folder= CHECKPOINT_PATH + "new_unnamed_bot" if (len(sys.argv) < 2) else (CHECKPOINT_PATH + sys.argv[1]),
                      add_unix_timestamp= (len(sys.argv) < 2)
                      )
    
    if len(sys.argv) == 2:
        # load from folder and use newest checkpoint
        if(bot_exists(sys.argv[1])):
            learner.load(CHECKPOINT_PATH + sys.argv[1] + "/" + find_newest_checkpoint(CHECKPOINT_PATH + sys.argv[1]), LOG_TO_WANDB)
    elif len(sys.argv) == 3:
        # load from folder and use specific checkpoint
        if(bot_exists(sys.argv[1])):
            learner.load(CHECKPOINT_PATH + sys.argv[1]  + "/" + sys.argv[2], LOG_TO_WANDB)

    spawn_opponents = False
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine = (VelocityPlayerToBallReward(),
                          VelocityBallToGoalReward(),
                          EventReward(team_goal=1, concede=-1, demo=0.1, touch=0.2, shot=0.4, save=0.5))
    reward_weights = (0.01, 0.1, 10.0)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    state_setter = CustomState(ball_on_ground=True)

    env = rlgym_sim.make(tick_skip=tick_skip,
                            team_size=team_size,
                            spawn_opponents=spawn_opponents,
                            terminal_conditions=terminal_conditions,
                            reward_fn=reward_fn,
                            obs_builder=obs_builder,
                            action_parser=action_parser,
                            state_setter=state_setter)

    episodes = 100

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # inference_batch = np.concatenate(obs, axis=0)
            actions, log_probs = learner.ppo_learner.policy.get_action(obs=obs)
            actions = actions.numpy().astype(np.float32)
            # print(f"The actions are \n\n{actions}")
            env.render()
            obs, reward, done, info = env.step(actions)
            time.sleep(1/30)