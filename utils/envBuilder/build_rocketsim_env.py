from utils.Rewards.firstReward import firstReward
from utils.Rewards.sparseReward import sparseReward
from utils.Rewards.defaultReward import defaultReward
from utils.stateSetter.stateSetter import CustomState
import rlgym_sim
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.state_setters import RandomState
import numpy as np

# DO NOT CALL THE FUNCTION HERE!!! ##########################################################
reward_fn = defaultReward

def build_rocketsim_env():    
    spawn_opponents = False
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]


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
                         reward_fn=reward_fn(),
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    return env