"""Episode and aggregate metrics for REAL world-graph tasks."""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodeStats:
    episode_id: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    target_diff_count: int = 0
    completed_diff_count: int = 0
    completion_rate: float = 0.0
    is_success: bool = False
    is_timeout: bool = False
    is_invalid_action: bool = False

    matched_diffs: List[Tuple[str, str, str]] = field(default_factory=list)
    wrong_diffs: List[Tuple[str, str, str]] = field(default_factory=list)


class MCPDiffMetric:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.success_threshold = self.config.get("success_threshold", 1.0)

        self.episode_stats: List[EpisodeStats] = []
        self.current_episode: Optional[EpisodeStats] = None

        self.total_episodes = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_invalid_actions = 0

        self.window_size = self.config.get("window_size", 100)

    def reset_episode(self, episode_id: int = None, target_diff_count: int = 0):
        if episode_id is None:
            episode_id = self.total_episodes

        self.current_episode = EpisodeStats(episode_id=episode_id, target_diff_count=target_diff_count)

    def update_step(self, reward: float, info: Dict[str, Any], terminated: bool = False, truncated: bool = False):
        if self.current_episode is None:
            return

        self.current_episode.total_steps += 1
        self.current_episode.total_reward += reward

        if "matched_diffs" in info:
            self.current_episode.matched_diffs.extend(info["matched_diffs"])
            self.current_episode.completed_diff_count += len(info["matched_diffs"])

        if "wrong_diffs" in info:
            self.current_episode.wrong_diffs.extend(info["wrong_diffs"])

        if "completion_rate" in info:
            self.current_episode.completion_rate = info["completion_rate"]

        if "perfect_match" in info and info["perfect_match"]:
            self.current_episode.is_success = True

    def end_episode(
        self, success: bool = None, timeout: bool = False, invalid_action: bool = False, completion_rate: float = None
    ):
        if self.current_episode is None:
            return

        if completion_rate is not None:
            self.current_episode.completion_rate = completion_rate
        elif self.current_episode.target_diff_count > 0:
            self.current_episode.completion_rate = (
                self.current_episode.completed_diff_count / self.current_episode.target_diff_count
            )

        if success is not None:
            self.current_episode.is_success = success
        else:
            self.current_episode.is_success = self.current_episode.completion_rate >= self.success_threshold

        self.current_episode.is_timeout = timeout
        self.current_episode.is_invalid_action = invalid_action

        self.total_episodes += 1
        if self.current_episode.is_success:
            self.total_successes += 1
        if timeout:
            self.total_timeouts += 1
        if invalid_action:
            self.total_invalid_actions += 1

        self.episode_stats.append(self.current_episode)

        max_history = self.config.get("max_history", 10000)
        if len(self.episode_stats) > max_history:
            self.episode_stats = self.episode_stats[-max_history:]

        self.current_episode = None

    def get_metrics(self) -> Dict[str, float]:
        if self.total_episodes == 0:
            return {
                "success_rate": 0.0,
                "avg_completion_rate": 0.0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
                "timeout_rate": 0.0,
                "invalid_action_rate": 0.0,
                "total_episodes": 0,
            }

        metrics = {
            "success_rate": self.total_successes / self.total_episodes,
            "timeout_rate": self.total_timeouts / self.total_episodes,
            "invalid_action_rate": self.total_invalid_actions / self.total_episodes,
            "total_episodes": self.total_episodes,
        }

        if self.episode_stats:
            completion_rates = [ep.completion_rate for ep in self.episode_stats]
            rewards = [ep.total_reward for ep in self.episode_stats]
            steps = [ep.total_steps for ep in self.episode_stats]

            metrics["avg_completion_rate"] = np.mean(completion_rates)
            metrics["avg_reward"] = np.mean(rewards)
            metrics["avg_steps"] = np.mean(steps)
            metrics["std_completion_rate"] = np.std(completion_rates)
            metrics["std_reward"] = np.std(rewards)
            metrics["std_steps"] = np.std(steps)

        return metrics

    def get_recent_metrics(self, n: int = None) -> Dict[str, float]:
        n = n or self.window_size
        recent_episodes = self.episode_stats[-n:] if self.episode_stats else []

        if not recent_episodes:
            return self.get_metrics()

        recent_successes = sum(1 for ep in recent_episodes if ep.is_success)
        recent_timeouts = sum(1 for ep in recent_episodes if ep.is_timeout)
        recent_invalid = sum(1 for ep in recent_episodes if ep.is_invalid_action)

        completion_rates = [ep.completion_rate for ep in recent_episodes]
        rewards = [ep.total_reward for ep in recent_episodes]
        steps = [ep.total_steps for ep in recent_episodes]

        return {
            "recent_success_rate": recent_successes / len(recent_episodes),
            "recent_avg_completion_rate": np.mean(completion_rates),
            "recent_avg_reward": np.mean(rewards),
            "recent_avg_steps": np.mean(steps),
            "recent_timeout_rate": recent_timeouts / len(recent_episodes),
            "recent_invalid_action_rate": recent_invalid / len(recent_episodes),
            "recent_episodes": len(recent_episodes),
        }

    def get_summary(self) -> str:
        metrics = self.get_metrics()
        recent = self.get_recent_metrics()

        lines = [
            "=" * 50,
            "MCP Diff Metric Summary",
            "=" * 50,
            f"Total Episodes: {metrics['total_episodes']}",
            "",
            "Overall Performance:",
            f"  Success Rate:     {metrics['success_rate']:.1%}",
            f"  Avg Completion:   {metrics.get('avg_completion_rate', 0):.1%}",
            f"  Avg Reward:       {metrics.get('avg_reward', 0):.2f}",
            f"  Avg Steps:        {metrics.get('avg_steps', 0):.1f}",
            "",
            "Failure Analysis:",
            f"  Timeout Rate:     {metrics['timeout_rate']:.1%}",
            f"  Invalid Action:   {metrics['invalid_action_rate']:.1%}",
            "",
            f"Recent {recent['recent_episodes']} Episodes:",
            f"  Success Rate:     {recent['recent_success_rate']:.1%}",
            f"  Avg Completion:   {recent['recent_avg_completion_rate']:.1%}",
            f"  Avg Reward:       {recent['recent_avg_reward']:.2f}",
            "=" * 50,
        ]

        return "\n".join(lines)

    def log_episode(self, episode: EpisodeStats = None) -> str:
        if episode is None:
            if not self.episode_stats:
                return "No episodes recorded."
            episode = self.episode_stats[-1]

        status = "SUCCESS" if episode.is_success else "FAILED"
        if episode.is_timeout:
            status += " (timeout)"
        if episode.is_invalid_action:
            status += " (invalid action)"

        lines = [
            f"Episode {episode.episode_id}: {status}",
            f"  Steps: {episode.total_steps}, Reward: {episode.total_reward:.2f}",
            f"  Completion: {episode.completed_diff_count}/{episode.target_diff_count} ({episode.completion_rate:.1%})",
        ]

        if episode.matched_diffs:
            lines.append(f"  Matched: {len(episode.matched_diffs)} diffs")

        if episode.wrong_diffs:
            lines.append(f"  Wrong: {len(episode.wrong_diffs)} diffs")

        return "\n".join(lines)

    def reset_all(self):
        self.episode_stats = []
        self.current_episode = None
        self.total_episodes = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_invalid_actions = 0


def create_metric(config: Dict = None) -> MCPDiffMetric:
    return MCPDiffMetric(config)
