from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward

def firstReward():
    rewards_to_combine = (VelocityPlayerToBallReward(),
                            VelocityBallToGoalReward(),
                            EventReward(team_goal=1, concede=-1, demo=0.1, touch=0.2, shot=0.4, save=0.5))
    reward_weights = (0.01, 0.1, 10.0)

    return CombinedReward(reward_functions=rewards_to_combine,
                            reward_weights=reward_weights)