from typing import Any, Dict, Union

import habitat
import numpy as np
import os
import sys
from habitat import Config, Dataset
from habitat.core.embodied_task import Metrics
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions.pose_utils import get_pose_change, get_sim_location


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@baseline_registry.register_env(name="VLNCEZeroShotEnv")
class VLNCEZeroShotEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Union[Dataset,None]=None) -> None:
        super().__init__(config.TASK_CONFIG, dataset)
        self.sensor_pose_sensor = self.habitat_env.task.sensor_suite.get('sensor_pose')
    
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

    def _ssa_set_agent_pose(self, position, yaw=None):
        sim = self._env.sim
        previous_sim_location = self.sensor_pose_sensor.last_sim_location
        init_state = sim.get_agent_state()
        rotation = init_state.rotation
        if yaw is not None:
            angle = float(yaw) + np.pi
            rotation = np.quaternion(np.cos(angle / 2.0), 0, np.sin(angle / 2.0), 0)
        sim.set_agent_state(np.asarray(position, dtype=np.float32), rotation)
        dx, dy, do, current_sim_location = get_pose_change(sim, previous_sim_location)
        self.sensor_pose_sensor.last_sim_location = current_sim_location
        self.sensor_pose_sensor.sensor_pose = [dx, dy, do]
        observations = sim.get_sensor_observations()
        observations["sensor_pose"] = np.asarray([dx, dy, do], dtype=np.float32)
        return observations

    def change_current_path(self, new_path: Any, collisions: Any):
        if 'current_path' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['current_path'] = [np.array(self._env.current_episode.start_position)]
        self._env.current_episode.info['current_path'] += new_path
        if 'collisions' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['collisions'] = []
        self._env.current_episode.info['collisions'] += collisions

    def _ssa_previous_step_collided(self) -> bool:
        sim = getattr(self._env, "sim", None)
        value = getattr(sim, "previous_step_collided", False)
        return bool(value() if callable(value) else value)

    def _ssa_adjust_episode_step_count(self, delta: int):
        env = self._env
        env._elapsed_steps = max(0, int(getattr(env, "_elapsed_steps", 0)) + int(delta))
        measures = getattr(getattr(env, "task", None), "measurements", None)
        step_measure = getattr(measures, "measures", {}).get("steps_taken") if measures is not None else None
        if step_measure is not None and hasattr(step_measure, "_metric"):
            step_measure._metric = max(0.0, float(step_measure._metric) + float(delta))
        task_active = bool(getattr(getattr(env, "task", None), "is_episode_active", True))
        if not task_active:
            env._episode_over = True
        elif env._max_episode_steps > 0:
            env._episode_over = env._elapsed_steps >= env._max_episode_steps
        return {
            "elapsed_steps": int(env._elapsed_steps),
            "episode_over": bool(env._episode_over),
        }
