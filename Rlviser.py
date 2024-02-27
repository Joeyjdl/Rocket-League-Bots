import time

import rlviser_py as vis
import RocketSim as rs
import Training as functions
import time

import rlgym_sim  
from rlgym_sim.utils.action_parsers import ContinuousAction
import time
import numpy as np
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
    EventReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction

# ## Load model
if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = functions.ExampleLogger()

    if len(functions.sys.argv) < 2:
        print("Please enter a name for the model you would like to create/load")
        exit(-1)

    # 32 processes
    n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(functions.build_rocketsim_env,
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
                      log_to_wandb=functions.LOG_TO_WANDB,
                      checkpoints_save_folder= functions.CHECKPOINT_PATH + "new_unnamed_bot" if (len(functions.sys.argv) < 2) else (functions.CHECKPOINT_PATH + functions.sys.argv[1]),
                      add_unix_timestamp= (len(functions.sys.argv) < 2)
                      )
    
    if len(functions.sys.argv) == 2:
        # load from folder and use newest checkpoint
        if(functions.bot_exists(functions.sys.argv[1])):
            learner.load(functions.CHECKPOINT_PATH + functions.sys.argv[1] + "/" + functions.find_newest_checkpoint(functions.CHECKPOINT_PATH + functions.sys.argv[1]), functions.LOG_TO_WANDB)
    elif len(functions.sys.argv) == 3:
        # load from folder and use specific checkpoint
        if(functions.bot_exists(functions.sys.argv[1])):
            learner.load(functions.CHECKPOINT_PATH + functions.sys.argv[1]  + "/" + functions.sys.argv[2], functions.LOG_TO_WANDB)

    spawn_opponents = True
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

    env = rlgym_sim.make(tick_skip=tick_skip,
                            team_size=team_size,
                            spawn_opponents=spawn_opponents,
                            terminal_conditions=terminal_conditions,
                            reward_fn=reward_fn,
                            obs_builder=obs_builder,
                            action_parser=action_parser)

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