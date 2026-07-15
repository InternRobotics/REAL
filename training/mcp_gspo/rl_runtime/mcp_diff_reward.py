"""World-graph difference rewards for REAL manipulation rollouts."""

from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy
from collections import Counter
import numpy as np


def normalize_object_name(obj_name: str) -> str:
    if "_on_" in obj_name:
        return obj_name.split("_on_")[0]
    return obj_name


class MCPDiffReward:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.initial_world_graph = None
        self.goal_world_graph = None

        self.target_diff_list: List[Tuple[str, str, str]] = []
        self.remaining_diffs: List[Tuple[str, str, str]] = []

        self.prev_world_graph = None
        self.prev_completion_rate = 0.0

        self.reward_weights = {
            "diff_match": 1.5,
            "perfect_match": 2.0,
            "step_penalty": -0.03,
            "step_penalty_start": 3,
            "wrong_diff": -0.5,
            "neutral_diff": 0.0,
            "progress_shaping": 2.0,
            "place_wrong_furniture": -2.0,
        }

        if "reward_weights" in self.config:
            self.reward_weights.update(self.config["reward_weights"])

        self.step_count = 0
        self.task_completed = False

    def reset(self, initial_wg: Dict, goal_wg: Dict):
        self.initial_world_graph = deepcopy(initial_wg)
        self.goal_world_graph = deepcopy(goal_wg)

        self.target_diff_list = self._compute_diff_list(self.initial_world_graph, self.goal_world_graph)
        self.remaining_diffs = self.target_diff_list.copy()

        self.prev_world_graph = deepcopy(self.initial_world_graph)
        self.prev_completion_rate = 0.0
        self.step_count = 0
        self.task_completed = False

        print(f"[MCPDiffReward] Reset: {len(self.target_diff_list)} target diffs to achieve")
        if len(self.target_diff_list) <= 10:
            for diff in self.target_diff_list:
                obj, furniture, op = diff
                print(f"  - {obj} {op} {furniture}")

    def compute_reward(self, current_wg: Dict, robot_inv: str = None, marker_map: Dict = None) -> Tuple[float, Dict]:
        self.step_count += 1

        current_step_diffs = self._compute_diff_list(self.prev_world_graph, current_wg)

        reward = 0.0
        info = {
            "matched_diffs": [],
            "wrong_diffs": [],
            "neutral_diffs": [],
            "remaining_count": len(self.remaining_diffs),
            "completion_rate": self.get_completion_rate(),
        }

        for diff in current_step_diffs:
            obj, furniture, op = diff

            if diff in self.remaining_diffs:
                info["matched_diffs"].append(diff)
                self.remaining_diffs.remove(diff)

                diff_reward = self.reward_weights["diff_match"]
                reward += diff_reward

                print(
                    f"[MCPDiffReward] ✓ Matched diff: {obj} {op} {furniture} "
                    f"({len(self.remaining_diffs)}/{len(self.target_diff_list)} remaining)"
                )

            else:
                reverse_op = "remove" if op == "add" else "add"
                reverse_diff = (obj, furniture, reverse_op)

                if reverse_diff in self.target_diff_list:
                    info["wrong_diffs"].append(diff)
                    wrong_penalty = self.reward_weights["wrong_diff"]
                    reward += wrong_penalty

                    print(f"[MCPDiffReward] ✗ Wrong diff (reverse): {obj} {op} {furniture}")
                else:
                    info["neutral_diffs"].append(diff)
                    neutral_reward = self.reward_weights["neutral_diff"]
                    reward += neutral_reward

                    if op == "add":
                        target_add_furnitures = [
                            f for o, f, operation in self.target_diff_list if o == obj and operation == "add"
                        ]
                        if target_add_furnitures:
                            place_wrong_penalty = self.reward_weights["place_wrong_furniture"]
                            reward += place_wrong_penalty
                            info["place_wrong_furniture"] = (obj, furniture, target_add_furnitures[0])
                            print(
                                f"[MCPDiffReward] ✗ Place to wrong furniture: {obj} -> {furniture} "
                                f"(should be {target_add_furnitures[0]}, penalty: {place_wrong_penalty})"
                            )

        if len(self.remaining_diffs) == 0 and not self.task_completed:
            perfect_reward = self.reward_weights["perfect_match"]
            reward += perfect_reward
            info["perfect_match"] = True
            self.task_completed = True
            print(f"[MCPDiffReward] 🎉 Task completed! All {len(self.target_diff_list)} diffs achieved.")

        # reward += w * (completion_rate_t - completion_rate_{t-1})

        current_completion_rate = self.get_completion_rate()
        progress_delta = current_completion_rate - self.prev_completion_rate
        if progress_delta != 0:
            progress_reward = self.reward_weights["progress_shaping"] * progress_delta
            reward += progress_reward
            info["progress_reward"] = progress_reward
            if progress_delta > 0:
                print(
                    f"[MCPDiffReward] 📈 Progress: {self.prev_completion_rate:.1%} -> {current_completion_rate:.1%} (+{progress_reward:.3f})"
                )
        self.prev_completion_rate = current_completion_rate

        step_penalty_start = self.reward_weights.get("step_penalty_start", 10)
        if self.step_count > step_penalty_start:
            step_penalty = self.reward_weights["step_penalty"]
            reward += step_penalty

        self.prev_world_graph = deepcopy(current_wg)

        if self.step_count % 20 == 0 or len(self.remaining_diffs) == 0:
            completion = self.get_completion_rate()
            print(f"[MCPDiffReward] Step {self.step_count}, Completion: {completion:.1%}, Reward: {reward:.4f}")

        info["total_reward"] = reward
        info["remaining_count"] = len(self.remaining_diffs)
        info["completion_rate"] = self.get_completion_rate()

        return reward, info

    def _compute_diff_list(self, wg_from: Dict, wg_to: Dict) -> List[Tuple[str, str, str]]:
        diff_list = []

        all_furniture = set(wg_from.keys()) | set(wg_to.keys())

        for furniture in all_furniture:
            from_objects = Counter(
                normalize_object_name(obj) for obj in wg_from.get(furniture, {}).get("content", []) if obj is not None
            )
            to_objects = Counter(
                normalize_object_name(obj) for obj in wg_to.get(furniture, {}).get("content", []) if obj is not None
            )

            added_objects = to_objects - from_objects
            for obj, count in added_objects.items():
                for _ in range(count):
                    diff_list.append((obj, furniture, "add"))

            removed_objects = from_objects - to_objects
            for obj, count in removed_objects.items():
                for _ in range(count):
                    diff_list.append((obj, furniture, "remove"))

        return diff_list

    def get_completion_rate(self) -> float:
        if len(self.target_diff_list) == 0:
            return 1.0
        return 1.0 - (len(self.remaining_diffs) / len(self.target_diff_list))

    def get_remaining_diffs(self) -> List[Tuple[str, str, str]]:
        return self.remaining_diffs.copy()

    def get_target_diffs(self) -> List[Tuple[str, str, str]]:
        return self.target_diff_list.copy()

    def is_done(self) -> bool:
        return len(self.remaining_diffs) == 0


class MCPDiffRewardWithPartialCredit(MCPDiffReward):
    def __init__(self, config: Dict = None):
        super().__init__(config)

        self.partial_credit_weights = {
            "pick_correct_object": 0.3,
            "partial_move": 0.5,
            "see_target_object": 0.2,
            "ask_reward": 1.0,
            "replan_recover_pick": 1.0,
            # Disambiguation chain rewards
            "ask_when_ambiguous": 0.5,  # R1: ask called while distractor visible
            "used_marker_from_answer": 0.5,  # R2: pick used the exact marker from ask response
            "blind_pick_penalty": -0.15,  # R3: picked without asking when distractors present
        }

        if config and "partial_credit_weights" in config:
            self.partial_credit_weights.update(config["partial_credit_weights"])

        self.picked_objects = set()
        self.partial_diffs_completed = {}

        self.replan_wrong_pick_obj: Optional[str] = None
        self.replan_wrong_pick_origin: Optional[str] = None
        self.replan_wrong_pick_returned: bool = False
        self.replan_bonus_given: bool = False

        self.seen_target_categories = set()
        self.ask_count = 0
        self.max_ask_rewards = 3

        # Disambiguation chain tracking
        self.distractor_ask_count = 0  # R1: asks while distractor visible (max 3 bonus rewards)
        self.max_distractor_ask_rewards = 3

    def reset(self, initial_wg: Dict, goal_wg: Dict):
        super().reset(initial_wg, goal_wg)

        self.picked_objects = set()
        self.partial_diffs_completed = {}

        self.replan_wrong_pick_obj = None
        self.replan_wrong_pick_origin = None
        self.replan_wrong_pick_returned = False
        self.replan_bonus_given = False

        self.seen_target_categories = set()
        self.ask_count = 0
        self.distractor_ask_count = 0

        self.obj_to_target_diffs = {}
        for diff in self.target_diff_list:
            obj, furniture, op = diff
            if obj not in self.obj_to_target_diffs:
                self.obj_to_target_diffs[obj] = []
            self.obj_to_target_diffs[obj].append((furniture, op))

        self.target_object_categories = set(self.obj_to_target_diffs.keys())

    def compute_reward(self, current_wg: Dict, robot_inv: str = None, marker_map: Dict = None) -> Tuple[float, Dict]:

        prev_wg = deepcopy(self.prev_world_graph) if self.prev_world_graph is not None else None

        base_reward, info = super().compute_reward(current_wg, robot_inv, marker_map)

        current_step_diffs = []
        if prev_wg is not None:
            current_step_diffs = self._compute_diff_list(prev_wg, current_wg)

        if robot_inv is not None:
            partial_reward = self._compute_partial_credit_rewards(current_wg, robot_inv)
            base_reward += partial_reward
            info["partial_reward"] = partial_reward

            replan_reward = self._compute_replan_recovery_reward(current_wg, robot_inv, current_step_diffs)
            base_reward += replan_reward
            info["replan_reward"] = replan_reward

        return base_reward, info

    def _find_object_location(self, world_graph: Dict, obj_name: str) -> Optional[str]:
        for furniture, payload in world_graph.items():
            contents = payload.get("content", []) if isinstance(payload, dict) else []
            for obj in contents:
                if normalize_object_name(obj) == obj_name:
                    return furniture
        return None

    def _compute_replan_recovery_reward(
        self,
        current_wg: Dict,
        robot_inv: str,
        current_step_diffs: List[Tuple[str, str, str]],
    ) -> float:
        if self.replan_bonus_given:
            return 0.0

        reward = 0.0
        normalized_inv = normalize_object_name(robot_inv) if robot_inv else None
        target_objects = set(self.obj_to_target_diffs.keys())

        if normalized_inv and normalized_inv not in target_objects and self.replan_wrong_pick_obj is None:
            self.replan_wrong_pick_obj = normalized_inv

            origin = None
            for obj, furniture, op in current_step_diffs:
                if obj == normalized_inv and op == "remove":
                    origin = furniture
                    break
            if origin is None:
                origin = self._find_object_location(self.initial_world_graph, normalized_inv)

            self.replan_wrong_pick_origin = origin
            self.replan_wrong_pick_returned = False

            print(f"[MCPDiffReward] ↩️ Replan tracking start: wrong pick {normalized_inv}, origin={origin}")

        wrong_obj = self.replan_wrong_pick_obj
        if wrong_obj and not self.replan_wrong_pick_returned:
            placed_back = False

            if self.replan_wrong_pick_origin is not None:
                placed_back = (wrong_obj, self.replan_wrong_pick_origin, "add") in current_step_diffs
            else:
                placed_back = any(obj == wrong_obj and op == "add" for obj, _, op in current_step_diffs)

            if placed_back:
                self.replan_wrong_pick_returned = True
                print(f"[MCPDiffReward] ✅ Replan step: returned wrong object {wrong_obj}")

        if (
            self.replan_wrong_pick_obj is not None
            and self.replan_wrong_pick_returned
            and normalized_inv
            and normalized_inv in target_objects
            and normalized_inv != self.replan_wrong_pick_obj
        ):
            reward = self.partial_credit_weights["replan_recover_pick"]
            self.replan_bonus_given = True

            print(
                f"[MCPDiffReward] 🎯 Replan recovery success: wrong={self.replan_wrong_pick_obj} "
                f"-> correct={normalized_inv} (+{reward:.2f})"
            )

            self.replan_wrong_pick_obj = None
            self.replan_wrong_pick_origin = None
            self.replan_wrong_pick_returned = False

        return reward

    def _compute_partial_credit_rewards(self, current_wg: Dict, robot_inv: str) -> float:
        reward = 0.0

        if robot_inv and robot_inv not in self.picked_objects:
            if robot_inv in self.obj_to_target_diffs:
                target_removes = [(furn, op) for furn, op in self.obj_to_target_diffs[robot_inv] if op == "remove"]

                if target_removes:
                    reward += self.partial_credit_weights["pick_correct_object"]
                    self.picked_objects.add(robot_inv)
                    print(
                        f"[MCPDiffReward] ✓ Picked target object: {robot_inv} "
                        f"(+{self.partial_credit_weights['pick_correct_object']:.2f})"
                    )

        for obj in self.obj_to_target_diffs:
            if obj in self.partial_diffs_completed:
                continue

            target_ops = self.obj_to_target_diffs[obj]
            remove_ops = [(f, op) for f, op in target_ops if op == "remove"]
            add_ops = [(f, op) for f, op in target_ops if op == "add"]

            if remove_ops and add_ops:
                remove_furniture = remove_ops[0][0]
                add_furniture = add_ops[0][0]

                current_obj_location = None
                for furniture, payload in current_wg.items():
                    if obj in payload.get("content", []):
                        current_obj_location = furniture
                        break

                if current_obj_location != remove_furniture and current_obj_location != add_furniture:
                    if obj not in self.partial_diffs_completed:
                        reward += self.partial_credit_weights["partial_move"]
                        self.partial_diffs_completed[obj] = ["remove"]
                        print(
                            f"[MCPDiffReward] ⚡ Partial move: {obj} removed from {remove_furniture} "
                            f"(+{self.partial_credit_weights['partial_move']:.2f})"
                        )

        return reward

    def compute_tool_reward(
        self, tool_name: str, tool_args: Dict, observation: str, disambiguation_state: Dict = None
    ) -> Tuple[float, Dict]:
        """
        Calculate tool-call rewards including disambiguation chain (R1/R2/R3).

        disambiguation_state keys:
            distractor_visible (bool): multiple target-category objects were seen
            last_ask_marker (str|None): marker_id returned by most recent ask
            last_ask_target (str|None): described target returned by most recent ask
            asked_since_ambiguity (bool): ask was called after distractor was detected
        """
        reward = 0.0
        info = {}
        dis = disambiguation_state or {}

        if tool_name == "ask":
            # Base ask reward (unchanged, max 3 times)
            reward, info = self._compute_ask_reward()

            # R1: bonus for asking specifically when distractors are present
            if dis.get("distractor_visible") and self.distractor_ask_count < self.max_distractor_ask_rewards:
                bonus = self.partial_credit_weights.get("ask_when_ambiguous", 0.5)
                reward += bonus
                self.distractor_ask_count += 1
                info["ask_when_ambiguous"] = bonus
                print(
                    f"[MCPDiffReward] 🎯 Ask-when-ambiguous bonus +{bonus:.2f} "
                    f"({self.distractor_ask_count}/{self.max_distractor_ask_rewards})"
                )

        elif tool_name == "pick":
            last_marker = dis.get("last_ask_marker")
            last_target = dis.get("last_ask_target")
            pick_marker = tool_args.get("marker_id", "")
            marker_map = dis.get("current_marker_map") or {}

            # R2: reward for using the exact marker the social agent specified
            if last_marker and pick_marker and pick_marker == last_marker:
                bonus = self.partial_credit_weights.get("used_marker_from_answer", 0.5)
                reward += bonus
                info["used_marker_from_answer"] = bonus
                print(f"[MCPDiffReward] ✅ Used marker from answer: {pick_marker} +{bonus:.2f}")

            elif last_target and pick_marker:
                marker_payload = marker_map.get(str(pick_marker), {}) if isinstance(marker_map, dict) else {}
                if isinstance(marker_payload, dict) and marker_payload.get("mock_target") == last_target:
                    bonus = self.partial_credit_weights.get("used_marker_from_answer", 0.5)
                    reward += bonus
                    info["used_target_from_answer"] = bonus
                    print(f"[MCPDiffReward] ✅ Used target description from answer: {last_target} +{bonus:.2f}")

            # R3: light penalty for picking blindly when distractors were visible and no ask followed
            elif dis.get("distractor_visible") and not dis.get("asked_since_ambiguity"):
                penalty = self.partial_credit_weights.get("blind_pick_penalty", -0.15)
                reward += penalty
                info["blind_pick_penalty"] = penalty
                print(f"[MCPDiffReward] ⚠️  Blind pick with distractors: {penalty:.2f}")

        return reward, info

    def _compute_see_target_reward(self, target_category: str, observation: str) -> Tuple[float, Dict]:
        reward = 0.0
        info = {"tool": "find_objects", "category": target_category}

        if target_category not in self.target_object_categories:
            info["matched"] = False
            info["reason"] = "not_target_category"
            return reward, info

        if target_category in self.seen_target_categories:
            info["matched"] = False
            info["reason"] = "already_rewarded"
            return reward, info

        negative_indicators = ["no object", "not found", "cannot find", "no target", "empty"]
        obs_lower = observation.lower()
        if any(neg in obs_lower for neg in negative_indicators):
            info["matched"] = False
            info["reason"] = "object_not_visible"
            return reward, info

        reward = self.partial_credit_weights["see_target_object"]
        self.seen_target_categories.add(target_category)
        info["matched"] = True
        info["reward"] = reward

        print(f"[MCPDiffReward] 👁️ Saw target object category: {target_category} (+{reward:.2f})")

        return reward, info

    def _compute_ask_reward(self) -> Tuple[float, Dict]:
        reward = 0.0
        info = {"tool": "ask", "ask_count": self.ask_count}

        if self.ask_count >= self.max_ask_rewards:
            info["rewarded"] = False
            info["reason"] = "max_ask_reached"
            return reward, info

        self.ask_count += 1
        reward = self.partial_credit_weights["ask_reward"]
        info["rewarded"] = True
        info["reward"] = reward
        info["ask_count"] = self.ask_count

        print(f"[MCPDiffReward] 💬 Ask tool used ({self.ask_count}/{self.max_ask_rewards}) (+{reward:.2f})")

        return reward, info


class MCPAntiExploitReward(MCPDiffRewardWithPartialCredit):
    """
    Anti-exploit reward function for MCP environment

    Extends MCPDiffRewardWithPartialCredit with additional anti-exploit mechanisms:
    1. One-time bonuses for state changes (target discovered, furniture reached)
    2. Penalties for wasteful actions (repeated perception, consecutive nav)
    3. Severe penalty for premature finish
    4. Prevents reward hacking through careful state tracking
    """

    def __init__(self, config: Dict = None):
        super().__init__(config)

        # Anti-exploit reward weights
        self.anti_exploit_weights = {
            "target_first_discovered": 1.0,  # Target object gets marker (first time)
            "first_visit_target_furniture": 0.3,  # First visit to target furniture
            "repeated_perception": -0.3,  # Perception without state change
            "consecutive_nav_same": -0.3,  # Navigate to same place repeatedly (legacy, kept for compat)
            "nav_revisit_window": -0.2,  # Revisit any nav target seen in last 6 steps
            "repeated_gaze": -0.2,  # Gaze at same marker repeatedly
            "premature_finish": -3.0,  # Finish before task complete
        }

        # Override from config
        if config and "anti_exploit_weights" in config:
            self.anti_exploit_weights.update(config["anti_exploit_weights"])

        # State tracking for anti-exploit
        self.target_objects_discovered = set()  # Track which target objects have been discovered
        self.visited_target_furnitures = set()  # Track which target furnitures visited
        self.last_perception_objects = set()  # Track objects from last perception action
        self.last_nav_target = None  # Track last navigation target (legacy)
        self.consecutive_nav_count = 0  # Count consecutive nav to same place (legacy)
        self.nav_history: list = []  # Sliding window of recent nav targets (for A→B→A detection)
        self.nav_window_size: int = 6  # Window size for nav revisit detection
        self.last_gaze_marker = None  # Track last gaze marker

    def reset(self, initial_wg: Dict, goal_wg: Dict):
        """Reset anti-exploit state tracking"""
        super().reset(initial_wg, goal_wg)

        # Reset anti-exploit state
        self.target_objects_discovered = set()
        self.visited_target_furnitures = set()
        self.last_perception_objects = set()
        self.last_nav_target = None
        self.consecutive_nav_count = 0
        self.nav_history = []
        self.last_gaze_marker = None

        # Extract target objects (unique hashes) and furnitures from diffs
        self.target_objects_set = set(obj for obj, furn, op in self.target_diff_list)
        self.target_furnitures_set = set(furn for obj, furn, op in self.target_diff_list)

        print(
            f"[AntiExploit] Target objects: {len(self.target_objects_set)}, "
            f"Target furnitures: {self.target_furnitures_set}"
        )

    def compute_reward(self, current_wg: Dict, robot_inv: str = None, marker_map: Dict = None) -> Tuple[float, Dict]:
        """Compute reward with anti-exploit mechanisms"""
        # Get base reward
        base_reward, info = super().compute_reward(current_wg, robot_inv, marker_map)

        # Compute anti-exploit rewards/penalties
        anti_exploit_reward = self._compute_anti_exploit_rewards(current_wg, robot_inv, marker_map)

        total_reward = base_reward + anti_exploit_reward
        info["anti_exploit_reward"] = anti_exploit_reward

        return total_reward, info

    def _compute_anti_exploit_rewards(self, current_wg: Dict, robot_inv: str, marker_map: Dict) -> float:
        """Compute anti-exploit rewards and penalties"""
        reward = 0.0

        # 1. Target object discovery reward (only once per object)
        if marker_map:
            current_marked_objects = set(marker_map.values()) if marker_map else set()

            # Check which target objects are newly discovered
            for target_obj in self.target_objects_set:
                if target_obj in self.target_objects_discovered:
                    continue  # Already rewarded

                # Check if this target object is now marked
                if any(target_obj in marked_obj for marked_obj in current_marked_objects):
                    reward += self.anti_exploit_weights["target_first_discovered"]
                    self.target_objects_discovered.add(target_obj)
                    print(
                        f"[AntiExploit] ✅ Target discovered: {target_obj[:20]}... "
                        f"(+{self.anti_exploit_weights['target_first_discovered']:.2f})"
                    )

        # Note: Navigation and perception tracking needs tool_name/args context
        # These will be tracked via a separate method called from env

        return reward

    def track_action(self, tool_name: str, tool_args: Dict, marker_map: Dict = None) -> float:
        """
        Track action-specific state for anti-exploit detection
        Called before compute_reward to track action context

        Returns immediate reward/penalty for this action
        """
        reward = 0.0

        # 2. Navigation tracking
        if tool_name in ("navigate_to", "nav_to"):
            nav_target = tool_args.get("receptacle_name")

            # Check if navigating to target furniture for first time
            if nav_target in self.target_furnitures_set:
                if nav_target not in self.visited_target_furnitures:
                    reward += self.anti_exploit_weights["first_visit_target_furniture"]
                    self.visited_target_furnitures.add(nav_target)
                    print(
                        f"[AntiExploit] ✅ First visit to target: {nav_target} "
                        f"(+{self.anti_exploit_weights['first_visit_target_furniture']:.2f})"
                    )

            # Sliding-window revisit detection: catches A→B→A→B loops that consecutive_nav_same misses.
            # Penalise each revisit proportionally to how many times it appears in the window.
            if nav_target in self.nav_history:
                revisit_count = self.nav_history.count(nav_target)
                penalty = self.anti_exploit_weights["nav_revisit_window"] * revisit_count
                reward += penalty
                print(f"[AntiExploit] ❌ Nav revisit x{revisit_count} in window: {nav_target} ({penalty:.2f})")

            self.nav_history.append(nav_target)
            if len(self.nav_history) > self.nav_window_size:
                self.nav_history.pop(0)

            # Keep legacy last_nav_target for any external callers
            self.last_nav_target = nav_target

        # 3. Perception tracking
        elif tool_name in [
            "find_objects",
            "explore_receptacle",
            "show_object_by_category",
            "walk_around",
        ]:
            # Track current visible objects from marker_map
            if marker_map:
                current_objects = set(marker_map.values()) if marker_map else set()

                # Check if perception changed anything
                if current_objects == self.last_perception_objects and self.last_perception_objects:
                    reward += self.anti_exploit_weights["repeated_perception"]
                    print(
                        f"[AntiExploit] ❌ Repeated perception (no change) "
                        f"({self.anti_exploit_weights['repeated_perception']:.2f})"
                    )

                self.last_perception_objects = current_objects

        # 4. Gaze tracking
        elif tool_name in ("focus_on", "gaze_at"):
            marker_id = tool_args.get("marker_id")

            if marker_id == self.last_gaze_marker:
                reward += self.anti_exploit_weights["repeated_gaze"]
                print(
                    f"[AntiExploit] ❌ Repeated gaze: marker {marker_id} "
                    f"({self.anti_exploit_weights['repeated_gaze']:.2f})"
                )

            self.last_gaze_marker = marker_id

        # 5. Premature finish (checked in env, but we track state here)
        elif tool_name == "finish":
            if not self.is_done():
                # This penalty will be applied by env, we just track for logging
                print(f"[AntiExploit] ⚠️ Premature finish detected (will penalize -3.0)")

        return reward
