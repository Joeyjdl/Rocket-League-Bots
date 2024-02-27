import time

import rlviser_py as vis
import RocketSim as rs
import Training as functions
import time

import rlgym_sim

TPS = 15

env = rlgym_sim.make(spawn_opponents=True)

while True:
    obs = env.reset()

    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    starttime = time.time()
    while not done:
        actions_1 = env.action_space.sample()
        actions_2 = env.action_space.sample()
        actions = [actions_1, actions_2]
        new_obs, reward, done, state = env.step(actions)
        env.render()
        ep_reward += reward[0]
        steps += 1

        # Sleep to keep the game in real time
        time.sleep(max(0, starttime + steps / TPS - time.time()))

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))

    

# ## Load model
# if __name__ == "__main__":
#     from rlgym_ppo import Learner
#     metrics_logger = functions.ExampleLogger()

#     if len(functions.sys.argv) < 2:
#         print("Please enter a name for the model you would like to create/load")
#         exit(-1)

#     # 32 processes
#     n_proc = 1

#     # educated guess - could be slightly higher or lower
#     min_inference_size = max(1, int(round(n_proc * 0.9)))

#     learner = Learner(functions.build_rocketsim_env,
#                       n_proc=n_proc,
#                       min_inference_size=min_inference_size,
#                       metrics_logger=metrics_logger,
#                       ppo_batch_size=50000,
#                       ts_per_iteration=50000,
#                       exp_buffer_size=150000,
#                       ppo_minibatch_size=50000,
#                       ppo_ent_coef=0.001,
#                       ppo_epochs=1,
#                       standardize_returns=True,
#                       standardize_obs=False,
#                       render=True,
#                       save_every_ts=100_000,
#                       timestep_limit=50_000_000_000,
#                       log_to_wandb=functions.LOG_TO_WANDB,
#                       checkpoints_save_folder= functions.CHECKPOINT_PATH + "new_unnamed_bot" if (len(functions.sys.argv) < 2) else (functions.CHECKPOINT_PATH + functions.sys.argv[1]),
#                       add_unix_timestamp= (len(functions.sys.argv) < 2)
#                       )
    
#     if len(functions.sys.argv) == 2:
#         # load from folder and use newest checkpoint
#         if(functions.bot_exists(functions.sys.argv[1])):
#             learner.load(functions.CHECKPOINT_PATH + functions.sys.argv[1] + "/" + functions.find_newest_checkpoint(functions.CHECKPOINT_PATH + functions.sys.argv[1]), functions.LOG_TO_WANDB)
#     elif len(functions.sys.argv) == 3:
#         # load from folder and use specific checkpoint
#         if(functions.bot_exists(functions.sys.argv[1])):
#             learner.load(functions.CHECKPOINT_PATH + functions.sys.argv[1]  + "/" + functions.sys.argv[2], functions.LOG_TO_WANDB)


#     learner.learn()



# game_mode = rs.GameMode.SOCCAR

# # Create example arena
# arena = rs.Arena(game_mode)

# # Set boost pad locations
# vis.set_boost_pad_locations([pad.get_pos().as_tuple() for pad in arena.get_boost_pads()])

# # Setup example arena
# car = arena.add_car(rs.Team.BLUE)
# car.set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50), boost=100))
# arena.ball.set_state(rs.BallState(pos=rs.Vec(y=400, z=100), ang_vel=rs.Vec(x=5)))
# car.set_controls(rs.CarControls(throttle=1, steer=1, boost=True))

# # Run for 3 seconds
# TIME = 3

# steps = 0
# start_time = time.time()
# for i in range(round(TIME * arena.tick_rate)):
#     arena.step(1)

#     # Render the current game state
#     pad_states = [pad.get_state().is_active for pad in arena.get_boost_pads()]
#     ball = arena.ball.get_state()
#     car_data = [
#         (car.id, car.team, car.get_config(), car.get_state())
#         for car in arena.get_cars()
#     ]

#     vis.render(steps, arena.tick_rate, game_mode, pad_states, ball, car_data)

#     # sleep to simulate running real time (it will run a LOT after otherwise)
#     time.sleep(max(0, start_time + steps / arena.tick_rate - time.time()))
#     steps += 1

# # Tell RLViser to exit
# print("Exiting...")
# vis.quit()