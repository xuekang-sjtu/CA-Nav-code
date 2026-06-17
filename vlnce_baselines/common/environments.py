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
from shared.ssa.rollout import execute_waypoint_rollout


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

    def get_observation_at(
        self,
        source_position,
        source_rotation,
        keep_agent_at_new_pose: bool = False,
    ):
        return self._env.sim.get_observations_at(
            source_position,
            source_rotation,
            keep_agent_at_new_pose,
        )

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
            "rollout_steps": list(plan.rollout_steps),
            "target_position": np.asarray(plan.target_position, dtype=np.float32).tolist(),
            "target_yaw_deg": float(plan.target_yaw_deg),
            "error": str(plan.error or ""),
            "planned_action_sequence": ["SSA"] * len(plan.rollout_steps),
            "planned_forward_actions": sum(1 for action in plan.actions if int(action) == 1),
            "planned_rollout_steps": int(len(plan.rollout_steps)),
            "start_pose": start_pose,
        }

    def ssa_reached_target(self, target_position, target_yaw_deg: float):
        return self._ssa_planner.reached_target(
            self._env,
            np.asarray(target_position, dtype=np.float32),
            float(target_yaw_deg),
        )

    def ssa_execute_plan(self, plan_result: Dict[str, Any]):
        return execute_waypoint_rollout(self, self._ssa_planner, plan_result)

    def ssa_sync_sensor_pose(self):
        """Sync SensorPoseSensor.last_sim_location to current sim pose after SSA takeover.

        After waypoint rollout teleports the agent, the sensor's last_sim_location
        is stale — the next envs.step() would compute a massive delta spanning the
        entire teleport distance, corrupting the post-SSA navigation step.
        """
        self.sensor_pose_sensor.last_sim_location = get_sim_location(self._env.sim)
        self.sensor_pose_sensor.sensor_pose = [0.0, 0.0, 0.0]

    def _ssa_previous_step_collided(self) -> bool:
        sim = getattr(self._env, "sim", None)
        value = getattr(sim, "previous_step_collided", False)
        return bool(value() if callable(value) else value)
