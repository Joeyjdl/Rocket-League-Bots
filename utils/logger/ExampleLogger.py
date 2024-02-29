from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState
import numpy as np

class ExampleLogger(MetricsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.previous_goal_count = 0

    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.blue_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            # [[velocity, rotation, score], [velocity, rotation, score], [velocity, rotation, score], [velocity, rotation, score]]
            avg_linvel += metric_array[0]
            current_goals = metric_array[2]

            # Reset score on new game
            if self.previous_goal_count > 0 and current_goals == 0:
                self.previous_goal_count = 0
            # Only count new goals
            if current_goals > self.previous_goal_count:
                self.previous_goal_count = current_goals
        
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps,
                  "Goals_scored": self.previous_goal_count}
        wandb_run.log(report)