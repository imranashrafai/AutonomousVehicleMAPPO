import random
import time
import os
# 👉 NEW imports
from pathlib import Path
import imageio

import numpy as np
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.policy.idm_policy import ManualControllableIDMPolicy
from collections import defaultdict
import numpy as np
import json

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)
envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)

class EnvCore(object):
    """
    Environment core wrapper around MetaDrive multi-agent env.
    Handles metrics collection and (in eval mode) MP4 video recording.
    """

    def __init__(self, args, config):
        self.args = args
        env_cls_name = args.env
        self.env = envs[env_cls_name](config)

        self.agent_num = self.args.num_agents
        self.obs_dim = list(self.env.observation_space.values())[0].shape[0]
        self.action_dim = list(self.env.action_space.values())[0].shape[0]
        self.action_space = list(self.env.action_space.values())
        self.observation_space = list(self.env.observation_space.values())
        self.need_reset = False
        self.agent_done_info = []
        self.metrics_data = defaultdict(list)
        self.reward_data = defaultdict(list)
        self.agent_done_info_data = []
        self.epoch = 0
        self.time_step = 0
        self.temp_time_step = 0
        self.start_step = 0

        # ---------- NEW: video recording state ----------
        self.record_video = bool(getattr(self.args, "eval", False))
        self.video_writer = None
        self.video_path = None

        if self.record_video:
            # Save eval videos in a fixed folder (same as your dashboard constant)
            video_dir = Path(r"D:\AutonomousVehicleFYP\evaluation_results")
            video_dir.mkdir(parents=True, exist_ok=True)

            time_str = time.strftime("%Y%m%d_%H%M%S")
            # Example: mappo_intersection_20241202_153000.mp4
            self.video_path = video_dir / f"{self.args.algorithm_name}_{self.args.env}_{time_str}.mp4"

            # Use H.264 for normal MP4 playback
            self.video_writer = imageio.get_writer(
                str(self.video_path),
                fps=20,
                codec="libx264"
            )
            print(f"[EnvCore] Evaluation video recording enabled: {self.video_path}")

    def reset(self):
        state = self.env.reset()
        sub_agent_obs = list(state[0].values())
        return sub_agent_obs

    def _record_frame_if_needed(self):
        """Grab a frame from the env and append to the video if in eval mode."""
        if not self.record_video or self.video_writer is None:
            return

        frame = self.env.render(mode="rgb_array")
        if isinstance(frame, dict):
            # MetaDrive can return dict of cameras; take main camera if present
            frame = frame.get("main_camera", list(frame.values())[0] if frame else None)

        if isinstance(frame, np.ndarray):
            self.video_writer.append_data(frame)

    def step(self, actions):
        self.time_step += 1
        self.temp_time_step += 1

        sub_agent_obs, reward, done, truncateds, info = self.env.step(
            {agent_id: action for agent_id, action in zip(self.env.vehicles.keys(), actions)}
        )
        sub_agent_obs = list(sub_agent_obs.values())
        sub_agent_reward = list(reward.values())
        sub_agent_done = list(done.values())[:-1]
        sub_agent_info = list(info.values())

        # ---------- NEW: record frame each env step during eval ----------
        self._record_frame_if_needed()

        # ------- your existing metrics logic below -------
        for agent_info in sub_agent_info:
            if any([
                agent_info.get("crash_vehicle", False),
                agent_info.get("crash_object", False),
                agent_info.get("crash_building", False),
                agent_info.get("crash_human", False),
                agent_info.get("crash_sidewalk", False),
                agent_info.get("out_of_road", False),
                agent_info.get("max_step", False),
                agent_info.get("crash", False),
                agent_info.get("arrive_dest", False)
            ]):
                self.agent_done_info.append(agent_info)

        new_agent_processed = False  # Flag to track if new agent's data is already used
        for agent_index, agent_done in enumerate(sub_agent_done):
            if agent_done:
                if len(sub_agent_done) > self.agent_num:
                    sub_agent_obs[agent_index] = sub_agent_obs[-1]
                    sub_agent_obs = np.delete(sub_agent_obs, -1, axis=0)
                    sub_agent_reward = np.delete(sub_agent_reward, -1)
                    sub_agent_done = np.delete(sub_agent_done, -1)
                    sub_agent_info = np.delete(sub_agent_info, -1)
                    new_agent_processed = True
                elif new_agent_processed and len(sub_agent_done) == self.agent_num:
                    continue
                else:
                    break

        while len(sub_agent_done) < self.agent_num:
            sub_agent_done = np.append(sub_agent_done, False)
            sub_agent_reward = np.append(sub_agent_reward, 0)
            sub_agent_obs.append(sub_agent_obs[-1])
            sub_agent_info = np.append(sub_agent_info, sub_agent_info[-1])
            self.need_reset = False

        if self.temp_time_step >= 5000:
            self.epoch += 1
            print(f"Epoch {self.epoch}: Information collection completed")
            total_agents = len(self.agent_done_info)
            crash_count = sum(1 for info in self.agent_done_info if info.get("crash", False))
            arrive_dest_count = sum(1 for info in self.agent_done_info if info.get("arrive_dest", False))
            total_episode_length = sum(info.get("episode_length", 0) for info in self.agent_done_info)
            success_episode_length = sum(
                info.get("episode_length", 0) for info in self.agent_done_info if info.get("arrive_dest", False)
            )

            safety_rate = (total_agents - crash_count) / total_agents if total_agents > 0 else 0
            success_rate = arrive_dest_count / total_agents if total_agents > 0 else 0

            if arrive_dest_count > 0:
                average_success_length = success_episode_length / arrive_dest_count
                time_efficiency = 1 - (average_success_length / 1000)
            else:
                time_efficiency = 0

            print(f"Safety Rate: {safety_rate:.2f}")
            print(f"Success Rate: {success_rate:.2f}")
            print(f"Time Efficiency: {time_efficiency:.2f}")
            num_vehicles = len(self.agent_done_info)
            print(f"Collected this round {num_vehicles} cars information")
            self.metrics_data[self.epoch] = [safety_rate, success_rate, time_efficiency]

            def convert_float32_to_float(obj):
                if isinstance(obj, dict):
                    return {k: convert_float32_to_float(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_float32_to_float(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_float32_to_float(item) for item in obj)
                elif isinstance(obj, np.float32):
                    return float(obj)
                else:
                    return obj

            converted_data = convert_float32_to_float(self.agent_done_info)
            self.agent_done_info_data.append({
                'start_step': self.start_step,
                'end_step': self.time_step,
                'agent_info': converted_data
            })

            total_reward = sum(info.get("episode_reward", 0) for info in self.agent_done_info)
            avg_reward = total_reward / len(self.agent_done_info)
            self.reward_data[self.epoch].append(avg_reward)

            output_dir = f"results/{self.args.env_name}/{self.args.scenario_name}/{self.args.algorithm_name}/check/{self.args.run_num}/logs"
            os.makedirs(output_dir, exist_ok=True)

            reward_file = os.path.join(output_dir, 'reward_data.json')
            with open(reward_file, 'w') as file:
                json.dump(dict(self.reward_data), file)

            metrics_file = os.path.join(output_dir, 'metrics_data.json')
            with open(metrics_file, 'w') as file:
                json.dump(dict(self.metrics_data), file)

            agent_done_info_file = os.path.join(output_dir, 'agent_done_info_data.json')
            with open(agent_done_info_file, 'w') as file:
                json.dump(self.agent_done_info_data, file, indent=4)

            self.agent_done_info.clear()
            self.temp_time_step = 0
            self.start_step = self.time_step
            self.need_reset = True

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def close(self):
        # close env
        self.env.close()
        # ---------- NEW: close writer ----------
        if self.video_writer is not None:
            self.video_writer.close()
            print(f"[EnvCore] Evaluation video saved to: {self.video_path}")

    def get_metrics_data(self):
        return self.metrics_data