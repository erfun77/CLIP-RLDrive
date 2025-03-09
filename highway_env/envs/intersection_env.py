from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import clip
from PIL import Image, ImageOps
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_weights = torch.load('C:/Users/erfan/Desktop/HighwayEnv/HighwayEnv-master/highway_env/envs/fine_tuned_clip_upper_layers(V5).pth', weights_only = True)
model_clip.load_state_dict(fine_tuned_weights)

last_obs = None
proceed = 0

# def obs_to_pil_image(obs_frame):
#     obs_frame_normalized = obs_frame / 12.0
#     obs_frame_uint8 = (obs_frame_normalized * 255).astype(np.uint8)
#     pil_image = Image.fromarray( obs_frame_uint8, mode='L')
#     padding = (40, 40, 40, 40) 
#     padded_pil_image = ImageOps.expand(pil_image, border=padding, fill='white')
#     return padded_pil_image

def obs_to_pil_image(obs_frame): 
    obs_frame_normalized = obs_frame / 12.0
    obs_frame_uint8 = (obs_frame_normalized * 255).astype(np.uint8)
    pil_image = Image.fromarray(obs_frame_uint8, mode='L')
    padding = (40, 40, 40, 40)  # Adjust padding size
    padded_pil_image = ImageOps.expand(pil_image, border=padding, fill='white')
    return padded_pil_image

def combine_obs_frames(obs_frames):
    pil_images = [obs_to_pil_image(obs_frame.T) for obs_frame in obs_frames] 
    
    combined_width = sum(img.width for img in pil_images) 
    max_height = max(img.height for img in pil_images) 

    combined_image = Image.new('L', (combined_width, max_height))
    
    x_offset = 0
    for img in pil_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined_image

directory_name = "temp_images"

# Create the directory if it doesn't exist
if not os.path.exists(directory_name):
    os.makedirs(directory_name)


def resize_combined_image(combined_image, target_size=(1200, 500)):
    resized_image = combined_image.resize(target_size, Image.Resampling.LANCZOS)  # Resize the combined image
    return resized_image


class IntersectionEnv(AbstractEnv):
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": True,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 13,  # [s]
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        # print('reward:', rewards)
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        global last_obs  
        obs, reward, terminated, truncated, info = super().step(action)
        ###########################################################################################

        for vehicle in self.road.vehicles:
            if isinstance(vehicle, MDPVehicle):
                ego_vehicle = vehicle
        print("speed:", ego_vehicle.speed/10) 

        if  last_obs is not None:
            latest_frame = last_obs[3].T  
            fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
            axes.imshow(last_obs[3].T, cmap=plt.get_cmap('gray'))
            
            image_path = os.path.join(directory_name, "stacked_obs.png")
            plt.savefig(image_path)
            pil_image = Image.open(image_path)
            preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model_clip.encode_image(preprocessed_image)
                text_features = model_clip.encode_text(clip.tokenize(["Slow down.", "Proceed.", "Speed up."]).to(device))
                
                logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
                predicted_instruction = ["Slow down.", "Proceed.", "Speed up."][logits_per_image.argmax().item()]

                print(f"Predicted instruction: {predicted_instruction}")
            

            if action == 0 and predicted_instruction == "Slow down.":
                reward += logits_per_image.max().item() + 0.15
            if action == 1 and predicted_instruction == "Proceed.":
                reward += logits_per_image.max().item()
            if action == 2 and predicted_instruction == "Speed up.":
                reward += logits_per_image.max().item() * 1.5
            if (action != 0) and predicted_instruction ==  "Slow down.":
                reward -= 2   
            if (action != 2) and predicted_instruction ==  "Speed up.":
                reward -= 2
            
            plt.show()
            plt.close(fig)
        
        # # # Update the stored observation to the current observation
        # # # print("reward:", reward)
        last_obs = obs


        ###########################################################################################
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(
                (f"o{ego_id % 4}", f"ir{ego_id % 4}", 0)
            )
            destination = self.config["destination"] or "o" + str(
                self.np_random.integers(1, 4)
            )
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(60 + 5 * self.np_random.normal(1), 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(
                longitudinal + 5 + self.np_random.normal() * position_deviation
            ),
            #################################################################################################################
            # speed=8 + self.np_random.normal() * speed_deviation,
            speed= 6.5 + self.np_random.normal() * speed_deviation,
            #################################################################################################################
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "il" in vehicle.lane_index[0]
            and "o" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config


class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "long_off",
                        "lat_off",
                        "ang_off",
                    ],
                },
                "action": {
                    "type": "ContinuousAction",
                    "steering_range": [-np.pi / 3, np.pi / 3],
                    "longitudinal": True,
                    "lateral": True,
                    "dynamical": True,
                },
            }
        )
        return config
