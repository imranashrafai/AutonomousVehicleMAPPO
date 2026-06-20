import os
import cv2
import metadrive.engine.asset_loader as asset_loader
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from stable_baselines3 import PPO  # Replace with your RL framework

# --- DISABLE FUSE MOUNT ---
asset_loader.DISABLE_MOUNT = True

MODEL_PATH = r"D:\AutonomousVehicleFYP\MAPPO_AVs\mappo\results\MyEnv\Intersection_MAPPO\rmappo\check\run1\models"
VIDEO_OUTPUT = r"D:\AutonomousVehicleFYP\MAPPO_AVs\mappo\videos\intersection.mp4"

NUM_AGENTS = 4
FPS = 30

env = MultiAgentIntersectionEnv(num_agents=NUM_AGENTS, traffic_density=0.2, use_render=True)

agents = []
for i in range(NUM_AGENTS):
    agent_file = os.path.join(MODEL_PATH, f"agent_{i}.zip")
    agents.append(PPO.load(agent_file, env=env))

frames = []
obs = env.reset()
done = [False] * NUM_AGENTS

while not all(done):
    actions = [agent.predict(obs[i], deterministic=True)[0] for i, agent in enumerate(agents)]
    obs, rewards, done, info = env.step(actions)
    frames.append(env.render(mode="rgb_array"))

env.close()

height, width, _ = frames[0].shape
video_writer = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
for frame in frames:
    video_writer.write(frame)
video_writer.release()
print(f"Video saved at {VIDEO_OUTPUT}")
