from typing import Any, Dict, Union

import habitat
import numpy as np
import os
import sys
from habitat import Config, Dataset
from habitat.core.embodied_task import Metrics
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.pose_utils import get_sim_location


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.ssa.planner import SimulatorAStarPlanner


@baseline_registry.register_env(name="VLNCEZeroShotEnv")
class VLNCEZeroShotEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Union[Dataset,None]=None) -> None:
        super().__init__(config.TASK_CONFIG, dataset)
        self.sensor_pose_sensor = self.habitat_env.task.sensor_suite.get('sensor_pose')
        self._ssa_planner = SimulatorAStarPlanner()
    
    def reset(self) -> Observations:
        self.sensor_pose_sensor.episode_start = False
        self.last_sim_location = get_sim_location(self.habitat_env.sim)
        self.sensor_pose = [0., 0., 0.] # initialize last sensor pose as [0,0,0]
        obs = super().reset()
        
        return obs
    
    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        obs, reward, done, info = super().step(action, **kwargs)
        
        return obs, reward, done, info

    def get_reward(self, observations: Observations) -> float:
        return 0.0
        
    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()
    
    def get_done(self, observations):
        return self._env.episode_over
    
    def get_reward_range(self):
        return (0.0, 0.0)
    
    def get_reward(self, observations: Observations) -> Any:
        return 0.0
    
    def _get_sensor_pose(self):
        pass

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        return {
            "position": agent_state.position.tolist(),
            "stop": self._env.task.is_stop_called,
        }

    def change_current_path(self, new_path: Any, collisions: Any):
        if 'current_path' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['current_path'] = [np.array(self._env.current_episode.start_position)]
        self._env.current_episode.info['current_path'] += new_path
        if 'collisions' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['collisions'] = []
        self._env.current_episode.info['collisions'] += collisions

    def ssa_build_plan(self, pose: Dict[str, Any]):
        start_pose = self._ssa_planner.current_pose(self._env)
        plan = self._ssa_planner.build_plan(self._env, pose)
        return {
            "actions": [int(action) for action in plan.actions],
            "target_position": np.asarray(plan.target_position, dtype=np.float32).tolist(),
            "target_yaw_deg": float(plan.target_yaw_deg),
            "error": str(plan.error or ""),
            "planned_action_sequence": [int(action) for action in plan.actions],
            "planned_forward_actions": sum(1 for action in plan.actions if int(action) == 1),
            "start_pose": start_pose,
        }

    def ssa_reached_target(self, target_position, target_yaw_deg: float):
        return self._ssa_planner.reached_target(
            self._env,
            np.asarray(target_position, dtype=np.float32),
            float(target_yaw_deg),
        )

    def ssa_execute_plan(self, plan_result: Dict[str, Any]):
        target_position = np.asarray(plan_result.get("target_position", []), dtype=np.float32)
        target_yaw_deg = float(plan_result.get("target_yaw_deg", 0.0) or 0.0)
        actions = [int(action) for action in plan_result.get("actions", []) or []]
        observations = self._env.sim.get_observations_at(
            self._env.sim.get_agent_state().position,
            self._env.sim.get_agent_state().rotation,
        )
        info = self.get_info(observations)
        success = False
        reason = str(plan_result.get("error", "") or "plan_exhausted")
        start_pose = plan_result.get("start_pose") or self._ssa_planner.current_pose(self._env)
        planned_action_sequence = [int(action) for action in actions]

        def finish(done, reason, success, actions_executed):
            result = {
                "observations": observations,
                "done": done,
                "info": info,
                "success": bool(success),
                "reason": str(reason),
                "actions_executed": int(actions_executed),
                "planned_action_sequence": planned_action_sequence,
                "start_pose": start_pose,
            }
            result.update(self._ssa_planner.pose_error(self._env, target_position, target_yaw_deg))
            return result

        for idx, action in enumerate(actions):
            prev_position = np.asarray(self._env.sim.get_agent_state().position, dtype=np.float32)
            observations = self._env.step(action)
            info = self.get_info(observations)
            done = self.get_done(observations)
            if done:
                reason = "episode_done"
                return finish(done, reason, False, idx + 1)
            if action == 1:
                curr_position = np.asarray(self._env.sim.get_agent_state().position, dtype=np.float32)
                if float(np.linalg.norm(curr_position - prev_position)) < 0.05:
                    reason = "forward_progress_failed"
                    return finish(done, reason, False, idx + 1)
            if self._ssa_planner.reached_target(self._env, target_position, target_yaw_deg):
                success = True
                reason = "reached_target"
                return finish(done, reason, success, idx + 1)
        done = self.get_done(observations)
        return finish(done, reason, success, len(actions))
