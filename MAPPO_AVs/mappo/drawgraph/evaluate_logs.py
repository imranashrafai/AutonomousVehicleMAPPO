# evaluate_logs.py
import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_agents = 0
    crash_count = 0
    arrive_dest_count = 0
    total_episode_length = 0
    total_episode_reward = 0

    for item in data:
        for agent_info in item['agent_info']:
            total_agents += 1
            if agent_info.get('crash', False):
                crash_count += 1
            if agent_info.get('arrive_dest', False):
                arrive_dest_count += 1
            total_episode_length += agent_info.get('episode_length', 0)
            total_episode_reward += agent_info.get('episode_reward', 0)

    if total_agents == 0:
        return 0, 0, 0, 0

    safety = 1 - (crash_count / total_agents)
    success_rate = arrive_dest_count / total_agents
    efficiency = arrive_dest_count / total_agents if total_agents > 0 else 0
    avg_reward = total_episode_reward / total_agents

    return safety, success_rate, efficiency, avg_reward

def plot_metrics(metrics_dict):
    labels = ['Safety', 'Success Rate', 'Efficiency', 'Avg Reward']
    x = np.arange(len(metrics_dict))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        values = [metrics_dict[env][i] for env in metrics_dict]
        ax.plot(x, values, marker='o', label=label)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_dict.keys())
    ax.set_ylabel("Metric Value")
    ax.set_title("Evaluation Metrics Comparison")
    ax.legend()
    ax.grid(True)
    plt.show()

def main():
    # Replace with your saved JSON log files from different models
    log_files = {
        "MAPPO64": "mappo64.json",
        "MAPPO256": "mappo256.json",
        "RMAPPO64": "rmappo64.json",
        "RMAPPO256": "rmappo256.json"
    }

    metrics_dict = {}
    for name, file in log_files.items():
        safety, success, efficiency, avg_reward = calculate_metrics(file)
        metrics_dict[name] = [safety, success, efficiency, avg_reward]
        print(f"{name}: Safety={safety:.2f}, Success={success:.2f}, Efficiency={efficiency:.2f}, Avg Reward={avg_reward:.2f}")

    plot_metrics(metrics_dict)

if __name__ == "__main__":
    main()
