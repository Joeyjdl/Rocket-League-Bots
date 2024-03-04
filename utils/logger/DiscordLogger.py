from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState
import numpy as np

class DiscordLogger(MetricsLogger):
    def __init__(self):
        self.blue_score = 0
        self.orange_score = 0
        self.logger_steps = 0
        self.prev_shots = 0
        self.prev_saves = 0
        self.touch_count = 0
        self.prev_count = 0

    def _collect_metrics(self, game_state: GameState) -> list:
        ball_stats = np.array(
            [
                # Ball speed
                np.linalg.norm(game_state.ball.linear_velocity), #index 0
                # Ball height
                game_state.ball.position[2], # index 1
            ]
        )

        p_stats = np.zeros(10)
        cur_shots = 0
        cur_saves = 0
        cur_count = 0
        for p in game_state.players:
            cur_shots += p.match_shots
            cur_saves += p.match_saves
            cur_count += p.ball_touched if ball_stats[1] > 300 and not p.on_ground else 0

        new_shots = cur_shots - self.prev_shots
        new_saves = cur_saves - self.prev_saves
        new_count = cur_count - self.prev_count
        self.prev_shots = cur_shots
        self.prev_saves = cur_saves
        self.prev_count = cur_count

        for p in game_state.players:
            touch_height = ball_stats[1] if p.ball_touched and not p.on_ground else 0
            multi_touch_count = new_count if new_count >= 2 else 0
            self.prev_count = 0 if new_count >= 2 else new_count
            p_stats += np.array(
                [
                    # Car speed
                    np.linalg.norm(p.car_data.linear_velocity), # index 2
                    # Car height
                    p.car_data.position[2], # index 3
                    # Boost held
                    float(p.boost_amount), # index 4
                    #reward 
                    # On ground
                    float(p.on_ground), #index 0
                    # Ball touch
                    float(p.ball_touched), #index 1
                    # Is demoed
                    float(p.is_demoed), #index 2
                    # Match shots
                    new_shots, #index 3
                    # Match saves
                    new_saves, #index 4
                    # Touch Height
                    touch_height,
                    multi_touch_count, #index 5
                ]
            )
        p_stats /= len(game_state.players)

        goal_scored = np.zeros(1) #index 6
        if (
            game_state.blue_score > self.blue_score
            or game_state.orange_score > self.orange_score
        ):
            goal_scored[0] = 1
        self.blue_score = game_state.blue_score
        self.orange_score = game_state.orange_score

        return np.concatenate([ball_stats, p_stats, goal_scored])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        step_diff = cumulative_timesteps - self.logger_steps
        self.logger_steps = cumulative_timesteps

        metrics = np.array(collected_metrics)

        mean_metrics = metrics[:, :5, :].mean(axis=0).mean(axis=1)

        rate_metrics = metrics[:, 5:, :].sum(axis=0).sum(axis=1) / step_diff

        report = {
            "ball_speed": mean_metrics[0],
            "ball_height": mean_metrics[1],
            "car_speed": mean_metrics[2],
            "car_height": mean_metrics[3],
            "boost_held": mean_metrics[4],
            "on_ground": rate_metrics[0],
            "touch_rate": rate_metrics[1],
            "demoed_rate": rate_metrics[2],
            "shot_rate": rate_metrics[3],
            "save_rate": rate_metrics[4],
            "avg_shot+save": (rate_metrics[3] + rate_metrics[4]) / 2,
            "touch_height": rate_metrics[5],
            "multiple_touch": rate_metrics[6],
            "goal_rate": rate_metrics[7],
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)