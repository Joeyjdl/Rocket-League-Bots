import numpy as np
from utils.logger.DiscordLogger import DiscordLogger
from utils.logger.ExampleLogger import ExampleLogger

import sys
import os
from datetime import datetime
from utils.envBuilder.build_rocketsim_env import build_rocketsim_env, reward_fn, clipParam

CHECKPOINT_PATH = "data/checkpoints/"
LOG_TO_WANDB = True

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

def bot_exists(name):
    folder = os.path.abspath("./data/checkpoints/" + name)
    return os.path.exists(folder)




if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = DiscordLogger()

    if len(sys.argv) < 2 or not sys.argv[1].isnumeric():
        print("Please enter a valid run number (check wanDB please)")
        exit(-1)
    
    if not reward_fn.__name__:
        print("You messed up something related to the reward function")
        exit(-1)

    print("##################################")
    print("Starting run for: " + reward_fn.__name__)
    print("Run number " + str(sys.argv[1]))
    print("##################################\n")
    run_name = reward_fn.__name__ + "_" + sys.argv[1]

    # 32 processes
    n_proc = 32
    if("--n_proc" in sys.argv):
        i = sys.argv.index("--n_proc")
        n_proc = int(sys.argv[i+1])
        print("n_proc=" + str(n_proc))

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
                      save_every_ts=10_000_000,
                      timestep_limit=1_000_000_000,
                      n_checkpoints_to_keep=100,
                      ppo_clip_range=clipParam,
                      log_to_wandb=LOG_TO_WANDB,
                      wandb_group_name="unnamed group" if (not reward_fn.__name__) else (reward_fn.__name__ + "_" + str(clipParam)) + "_" + sys.argv[1],
                      wandb_run_name=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                      wandb_project_name="rlgym-ppo",
                      checkpoints_save_folder= CHECKPOINT_PATH + ("new_unnamed_bot" if (len(sys.argv) < 2) else (run_name)),
                      add_unix_timestamp= (len(sys.argv) < 2),
                      random_seed = 123
                      )
    
    if len(sys.argv) == 2:
        # load from folder and use newest checkpoint
        if(bot_exists(run_name)):
            print("Loading bot " + run_name + " from newest checkpoint")
            learner.load(CHECKPOINT_PATH + run_name + "/" + find_newest_checkpoint(CHECKPOINT_PATH + run_name), LOG_TO_WANDB)
        else:
            print("The provided model, " + run_name + ", does not exist, creating new instead")
    elif len(sys.argv) == 3:
        # load from folder and use specific checkpoint
        print("Loading bot " + run_name + " from checkpoint " + str(sys.argv[2]))
        if(bot_exists(run_name)):
            learner.load(CHECKPOINT_PATH + run_name  + "/" + sys.argv[2], LOG_TO_WANDB)
    



    learner.learn()

