from rlgym_sim.utils.reward_functions.common_rewards import  EventReward

def rocketReward():
    return EventReward(goal=100, concede=-100, shot=10, save=50)