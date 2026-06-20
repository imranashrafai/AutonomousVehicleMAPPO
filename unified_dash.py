# unified_dashboard.py
import os
import base64
import streamlit as st
import sys
import time
import json
import traceback
from pathlib import Path
import glob
import subprocess
import threading
import imageio

# NEW: screen capture for evaluation video
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    mss = None
    MSS_AVAILABLE = False

# Try to insert VENV path early
try:
    VENV_PATH = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    if str(VENV_PATH) not in sys.path:
        sys.path.insert(0, str(VENV_PATH))
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TensorBoard imports
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# MetaDrive Imports
PANDA3D_AVAILABLE = False
MULTI_AGENT_CLASS = None
IDMPolicy = None

try:
    from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
    MULTI_AGENT_CLASS = MultiAgentMetaDrive
    PANDA3D_AVAILABLE = True
except Exception:
    try:
        from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDriveEnv
        MULTI_AGENT_CLASS = MultiAgentMetaDriveEnv
        PANDA3D_AVAILABLE = True
    except Exception:
        pass

try:
    from metadrive.policy.idm_policy import IDMPolicy
except Exception:
    IDMPolicy = None


# -------------------------
# Signup / Signin (File Handling)
# -------------------------
USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def signup_user(username, email, password):
    users = load_users()

    if username in users:
        return False, "User already exists"

    users[username] = {
        "email": email,
        "password": password
    }
    save_users(users)
    return True, "Signup successful. Please login."

def signin_user(username, password):
    users = load_users()

    if username in users and users[username]["password"] == password:
        return True
    return False


# -------------------------
# Paths & Configuration
# -------------------------
RESULTS_DIR_ROOT = Path(r"D:\AutonomousVehicleFYP\MAPPO_AVs\mappo\results")
CHECKPOINT_BASE = RESULTS_DIR_ROOT / "MyEnv" / "Intersection_MAPPO"
TRAIN_SCRIPT = Path(r"D:\AutonomousVehicleFYP\MAPPO_AVs\mappo\train\train.py")
PYTHON_EXE = r"C:\Users\QC\AppData\Local\Programs\Python\Python38\python.exe"
SCREENSHOT_ROOT = Path(r"D:\AutonomousVehicleFYP\FYP_Screenshots")
EVALUATION_OUTPUT = Path(r"D:\AutonomousVehicleFYP\evaluation_results")
EVALUATION_OUTPUT.mkdir(parents=True, exist_ok=True)
SCREENSHOT_ROOT.mkdir(parents=True, exist_ok=True)

WATERMARK_TEXT_DEFAULT = "Imran Ashraf, VLCMatrix Lab | FYP Documentation"

# -------------------------
# Utility Functions
# -------------------------
def load_font(img_h):
    """Dynamically loads Arial or fallback font."""
    size = max(14, int(img_h * 0.03))
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def add_watermark(path, text):
    """Adds watermark to image."""
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        font = load_font(h)
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        pad = max(10, int(h * 0.015))
        stroke = max(1, int(h * 0.003))
        
        outline_color = (0, 0, 0, 180)
        fill_color = (255, 255, 255, 220)

        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((pad + dx, pad + dy), text, font=font, fill=outline_color)
        
        draw.text((pad, pad), text, font=font, fill=fill_color)
        
        merged = Image.alpha_composite(img, txt_layer).convert("RGB")
        merged.save(path, quality=95)
    except Exception:
        st.error(f"Failed to apply watermark.")

def save_image(arr, out_path):
    """Saves numpy array (RGB) as PNG."""
    if isinstance(arr, np.ndarray):
        Image.fromarray(arr.astype(np.uint8)).save(out_path, quality=95)
    else:
        try:
            arr.save(out_path, quality=95)
        except Exception:
            raise

# -------------------------
# Model Evaluation Functions
# -------------------------
def find_all_checkpoints():
    """Find all available model checkpoints - finds all runs."""
    checkpoints = {
        'MAPPO': [],
        'RMAPPO': []
    }
    
    for algo in ['mappo', 'rmappo']:
        algo_path = CHECKPOINT_BASE / algo / "check"
        if algo_path.exists():
            run_folders = sorted(algo_path.glob("run*"), key=lambda x: x.name)
            
            for run_folder in run_folders:
                model_path = run_folder / "models"
                if model_path.exists() and (model_path / "actor.pt").exists():
                    checkpoints[algo.upper()].append({
                        'run': run_folder.name,
                        'path': str(model_path),
                        'algo': algo,
                        'full_path': str(run_folder)
                    })
    
    return checkpoints

def evaluate_model_sync(checkpoint_path, algo_name, render=True, num_agents=4):
    """
    Evaluate model using subprocess (synchronous).
    For MAPPO we must explicitly disable recurrent policy,
    otherwise train.py raises 'check recurrent policy!' AssertionError.
    """
    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--eval",
        "--eval_model_dir", checkpoint_path,
        "--algorithm_name", algo_name.lower(),
        "--num_agents", str(num_agents),
        "--scenario_name", "Intersection_MAPPO",
    ]

    if algo_name.lower() == "mappo":
        cmd.append("--use_recurrent_policy")

    if render:
        cmd.append("--use_render_eval")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(TRAIN_SCRIPT.parent),
            capture_output=True,
            text=True,
            timeout=600
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Evaluation timeout (10 minutes)",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }

def evaluate_model_and_record_video(checkpoint_path, algo_name, out_dir: Path, num_agents=4):
    """
    Run evaluation using trained MAPPO/RMAPPO model
    and capture the evaluation window from the screen into a GIF.
    """
    if not MSS_AVAILABLE:
        raise RuntimeError("The 'mss' package is required for recording. Install it with: pip install mss")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--eval",
        "--eval_model_dir", checkpoint_path,
        "--algorithm_name", algo_name.lower(),
        "--num_agents", str(num_agents),
        "--scenario_name", "Intersection_MAPPO",
        "--use_render_eval"
    ]

    if algo_name.lower() == "mappo":
        cmd.append("--use_recurrent_policy")

    proc = subprocess.Popen(
        cmd,
        cwd=str(TRAIN_SCRIPT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    video_frames = []

    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while proc.poll() is None:
            img = sct.grab(monitor)
            frame = Image.frombytes("RGB", img.size, img.rgb)
            video_frames.append(np.array(frame))
            time.sleep(1/15)

    stdout, stderr = proc.communicate()

    video_path = out_dir / "evaluation_simulation.gif"
    if video_frames:
        imageio.mimsave(video_path, video_frames, fps=15)

    return video_path, stdout.decode("utf-8", errors="ignore") if isinstance(stdout, bytes) else stdout, stderr.decode("utf-8", errors="ignore") if isinstance(stderr, bytes) else stderr

# -------------------------
# Simulation Runner
# -------------------------
def run_simulation_and_capture(map_code, num_agents, traffic_density, traffic_mode,
                               steps, snaps, out_dir: Path, watermark_text, seed):
    """Initializes MetaDrive environment, runs steps, captures screenshots, and records a GIF video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    if not MULTI_AGENT_CLASS:
        raise RuntimeError("Fatal: MultiAgentMetaDrive class not found. Check MetaDrive installation.")

    env_config = {
        "map": map_code,
        "num_agents": max(1, int(num_agents)),
        "start_seed": int(seed),
        "traffic_density": float(traffic_density),
        "traffic_mode": str(traffic_mode),
        "use_render": True,
        "horizon": max(steps, 1000),
        "agent_policy": IDMPolicy if traffic_density > 0.0 and IDMPolicy else "RandomPolicy",
    }

    try:
        env = MULTI_AGENT_CLASS(env_config)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate environment: {e}")

    try:
        env.reset()

        video_frames = []

        snap_steps = sorted(set([max(1, int((steps // (snaps + 1)) * (i + 1))) for i in range(snaps)]))
        next_snap_index = 0
        step = 0

        st_status = st.empty()
        st_progress = st.progress(0)

        while step < steps:
            actions = {agent_id: env.action_space.sample() for agent_id in env.agents.keys()}

            env.step(actions)

            frame = env.render(mode="rgb_array")

            if isinstance(frame, dict):
                frame = frame.get("main_camera", list(frame.values())[0] if frame else None)

            if isinstance(frame, np.ndarray):
                video_frames.append(frame)

                if next_snap_index < len(snap_steps) and step >= snap_steps[next_snap_index]:
                    fname = out_dir / f"{map_code}_step{step}.png"
                    Image.fromarray(frame.astype(np.uint8)).save(fname)
                    add_watermark(str(fname), watermark_text)
                    saved_files.append(str(fname))
                    next_snap_index += 1

            st_progress.progress(min(1.0, (step + 1) / steps))

            step += 1
            if all(env.dones.values()):
                break

        st_status.success(f"Simulation complete! Saved {len(saved_files)} screenshots.")

        if video_frames:
            video_path = out_dir / f"{map_code}_simulation.gif"
            imageio.mimsave(video_path, video_frames, fps=20)
            saved_files.append(str(video_path))
            st_status.success(f"Also saved video: {video_path.name}")

    finally:
        try:
            env.close()
        except Exception:
            pass

    return saved_files

# -------------------------
# TensorBoard Functions
# -------------------------
def load_individual_algorithm_metrics(algo_name):
    """Load and display all metrics for a single algorithm."""
    st.header(f"📊 {algo_name} Training Metrics")
    
    if not TENSORBOARD_AVAILABLE:
        st.error("TensorBoard not installed.")
        return
    
    base_scenario_path = RESULTS_DIR_ROOT / 'MyEnv/Intersection_MAPPO'
    algo_folder = base_scenario_path / algo_name.lower() / "check"
    
    if not algo_folder.exists():
        st.error(f"Path not found: {algo_folder}")
        return
    
    run_folders = sorted([p for p in algo_folder.glob('run*') if p.is_dir()])
    if not run_folders:
        st.error("No run folders found.")
        return
    
    metrics_data = {}
    
    with st.spinner(f"Loading {algo_name} metrics..."):
        for run_folder in run_folders:
            event_files = list(run_folder.rglob("events.out.tfevents.*"))
            if not event_files:
                continue
            
            for event_file in event_files:
                try:
                    event_acc = EventAccumulator(str(event_file))
                    event_acc.Reload()
                    
                    scalar_tags = event_acc.Tags().get('scalars', [])
                    if not scalar_tags:
                        continue
                    
                    for tag in scalar_tags:
                        if tag not in metrics_data:
                            metrics_data[tag] = {'steps': [], 'values': []}
                        
                        events = event_acc.Scalars(tag)
                        metrics_data[tag]['steps'].extend([e.step for e in events])
                        metrics_data[tag]['values'].extend([e.value for e in events])
                    
                except Exception:
                    continue
    
    if not metrics_data:
        st.error("No metrics data found.")
        return
    
    metric_names = sorted(list(metrics_data.keys()))
    st.info(f"Found {len(metric_names)} metrics for {algo_name}")
    
    smoothing = st.slider("Smoothing", 1, 50, 15)
    color = '#1f77b4' if algo_name == 'MAPPO' else '#ff7f0e'
    
    for i in range(0, len(metric_names), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            metric_name = metric_names[i]
            data = metrics_data[metric_name]
            df = pd.DataFrame({'x': data['steps'], 'y': data['values']}).sort_values('x')
            
            window = max(1, len(df) // smoothing)
            df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y_smooth'], mode='lines',
                name=algo_name, line=dict(width=2, color=color)
            ))
            
            fig.update_layout(
                title=metric_name,
                xaxis_title="Steps",
                yaxis_title="Value",
                height=350,
                template='plotly_white',
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if i + 1 < len(metric_names):
            with col2:
                metric_name = metric_names[i + 1]
                data = metrics_data[metric_name]
                df = pd.DataFrame({'x': data['steps'], 'y': data['values']}).sort_values('x')
                
                window = max(1, len(df) // smoothing)
                df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['x'], y=df['y_smooth'], mode='lines',
                    name=algo_name, line=dict(width=2, color=color)
                ))
                
                fig.update_layout(
                    title=metric_name,
                    xaxis_title="Steps",
                    yaxis_title="Value",
                    height=350,
                    template='plotly_white',
                    margin=dict(l=40, r=40, t=60, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

def load_and_plot_mappo_rewards():
    """Loads and plots rewards for MAPPO vs RMAPPO."""
    st.header("📈 Training Performance Analysis: MAPPO vs RMAPPO")
    
    if not TENSORBOARD_AVAILABLE:
        st.error("TensorBoard library required.")
        return
    
    base_scenario_path = RESULTS_DIR_ROOT / 'MyEnv/Intersection_MAPPO'
    
    algo_paths = {
        "MAPPO": base_scenario_path / "mappo" / "check",
        "RMAPPO": base_scenario_path / "rmappo" / "check",
    }
    
    reward_data = {}
    steps_data = {}

    with st.spinner("Loading training data..."):
        for name, base_folder in algo_paths.items():
            if not base_folder.exists():
                continue
                
            run_folders = sorted([p for p in base_folder.glob('run*') if p.is_dir()])
            if not run_folders:
                continue
            
            all_rewards = []
            all_steps = []
            
            for run_folder in run_folders:
                event_files = list(run_folder.rglob("events.out.tfevents.*"))
                if not event_files:
                    continue
                
                for event_file in event_files:
                    try:
                        event_acc = EventAccumulator(str(event_file))
                        event_acc.Reload()
                        
                        scalar_tags = event_acc.Tags().get('scalars', [])
                        if not scalar_tags:
                            continue
                        
                        reward_tags = [tag for tag in scalar_tags if 'reward' in tag.lower()]
                        if reward_tags:
                            events = event_acc.Scalars(reward_tags[0])
                            all_rewards.extend([e.value for e in events])
                            all_steps.extend([e.step for e in events])
                            break
                    except Exception:
                        continue
            
            if all_rewards:
                reward_data[name] = all_rewards
                steps_data[name] = all_steps

    if not reward_data:
        st.error("No training data found.")
        return

    smoothing = st.slider("Smoothing", 1, 50, 20)

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, (name, rewards) in enumerate(reward_data.items()):
        df = pd.DataFrame({'x': steps_data[name], 'y': rewards}).sort_values('x')
        window = max(1, len(rewards) // smoothing)
        df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y_smooth'], mode='lines',
            name=name, line=dict(width=3, color=colors[idx])
        ))
    
    fig.update_layout(
        title="MAPPO vs RMAPPO Training Performance",
        xaxis_title="Training Steps",
        yaxis_title="Episode Reward",
        height=500,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)'),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def load_and_plot_all_metrics():
    """Load and display all metrics for MAPPO and RMAPPO in grid layout."""
    st.header("📊 All Training Metrics: MAPPO vs RMAPPO")
    
    if not TENSORBOARD_AVAILABLE:
        st.error("TensorBoard library required.")
        return
    
    base_scenario_path = RESULTS_DIR_ROOT / 'MyEnv/Intersection_MAPPO'
    
    algo_paths = {
        "MAPPO": base_scenario_path / "mappo" / "check",
        "RMAPPO": base_scenario_path / "rmappo" / "check",
    }
    
    algo_metrics = {}
    
    with st.spinner("Loading all metrics..."):
        for algo_name, base_folder in algo_paths.items():
            if not base_folder.exists():
                continue
            
            run_folders = sorted([p for p in base_folder.glob('run*') if p.is_dir()])
            if not run_folders:
                continue
            
            metrics_data = {}
            
            for run_folder in run_folders:
                event_files = list(run_folder.rglob("events.out.tfevents.*"))
                if not event_files:
                    continue
                
                for event_file in event_files:
                    try:
                        event_acc = EventAccumulator(str(event_file))
                        event_acc.Reload()
                        
                        scalar_tags = event_acc.Tags().get('scalars', [])
                        if not scalar_tags:
                            continue
                        
                        for tag in scalar_tags:
                            if tag not in metrics_data:
                                metrics_data[tag] = {'steps': [], 'values': []}
                            
                            events = event_acc.Scalars(tag)
                            metrics_data[tag]['steps'].extend([e.step for e in events])
                            metrics_data[tag]['values'].extend([e.value for e in events])
                        
                    except Exception:
                        continue
            
            if metrics_data:
                algo_metrics[algo_name] = metrics_data
    
    if not algo_metrics:
        st.error("No metrics data found.")
        return
    
    all_metric_names = set()
    for algo_data in algo_metrics.values():
        all_metric_names.update(algo_data.keys())
    
    metric_names = sorted(list(all_metric_names))
    
    st.info(f"Found {len(metric_names)} unique metrics across both algorithms")
    
    with st.expander("📋 View Metric Details"):
        for algo_name, metrics in algo_metrics.items():
            st.write(f"**{algo_name}:** {len(metrics.keys())} metrics")
            st.write(", ".join(sorted(metrics.keys())))
    
    smoothing = st.slider("Smoothing", 1, 50, 15)
    
    for i in range(0, len(metric_names), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            metric_name = metric_names[i]
            fig = go.Figure()
            
            colors = {'MAPPO': '#1f77b4', 'RMAPPO': '#ff7f0e'}
            
            for algo_name in ['MAPPO', 'RMAPPO']:
                if algo_name in algo_metrics and metric_name in algo_metrics[algo_name]:
                    data = algo_metrics[algo_name][metric_name]
                    df = pd.DataFrame({'x': data['steps'], 'y': data['values']}).sort_values('x')
                    
                    window = max(1, len(df) // smoothing)
                    df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=df['x'], y=df['y_smooth'], mode='lines',
                        name=algo_name, line=dict(width=2, color=colors[algo_name])
                    ))
            
            fig.update_layout(
                title=metric_name,
                xaxis_title="Steps",
                yaxis_title="Value",
                height=350,
                template='plotly_white',
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if i + 1 < len(metric_names):
            with col2:
                metric_name = metric_names[i + 1]
                fig = go.Figure()
                
                colors = {'MAPPO': '#1f77b4', 'RMAPPO': '#ff7f0e'}
                
                for algo_name in ['MAPPO', 'RMAPPO']:
                    if algo_name in algo_metrics and metric_name in algo_metrics[algo_name]:
                        data = algo_metrics[algo_name][metric_name]
                        df = pd.DataFrame({'x': data['steps'], 'y': data['values']}).sort_values('x')
                        
                        window = max(1, len(df) // smoothing)
                        df['y_smooth'] = df['y'].rolling(window=window, min_periods=1).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=df['x'], y=df['y_smooth'], mode='lines',
                            name=algo_name, line=dict(width=2, color=colors[algo_name])
                        ))
                
                fig.update_layout(
                    title=metric_name,
                    xaxis_title="Steps",
                    yaxis_title="Value",
                    height=350,
                    template='plotly_white',
                    margin=dict(l=40, r=40, t=60, b=40),
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
                )
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Image Helper
# -------------------------
def get_img_as_base64(file_path):
    """Encodes a local file to a Base64 string for embedding in HTML."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def login_page():
    st.title("🔐 User Authentication")

    tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Sign Up"])

    with tab_login:
        st.subheader("Enter Credentials")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Submit Login"):
            with st.spinner("Authenticating..."):
                if signin_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please retry.")

    with tab_signup:
        st.subheader("Create New Account")

        new_username = st.text_input("Username", key="su_user")
        email = st.text_input("Email", key="su_email")
        new_password = st.text_input("Password", type="password", key="su_pass")

        if st.button("Sign Up"):
            if not new_username or not email or not new_password:
                st.error("All fields are required")
            else:
                success, msg = signup_user(new_username, email, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# -------------------------
# Main Streamlit App
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="MARL Dashboard - UOL", page_icon="🎓")
    
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if "username" not in st.session_state:
        st.session_state["username"] = ""

    if not st.session_state["logged_in"]:
        login_page()
        return

    st.sidebar.success(f"👤 {st.session_state.username}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
    
    # CSS Styling
    st.markdown("""
    <style>
    .uni-title {
        font-size: 2.2rem;
        font-weight: 900;
        text-align: center;
        color: #004d99;
        margin-bottom: 5px;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #222;
        margin-top: 30px;
        margin-bottom: 25px;
        text-align: center;
    }
    .team-card {
        background: white;
        padding: 24px 26px;
        border-radius: 15px;
        width: 320px;
        border: 2px solid #ddd;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        transition: 0.3s ease-in-out;
    }
    .team-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 28px rgba(0,0,0,0.15);
    }
    .profile-img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
        border: 3px solid #0066cc;
    }
    .team-title {
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 4px;
        color: #004d99;
    }
    .team-sub {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    .semester-info {
        font-size: 1.0rem;
        font-weight: bold;
        color: #cc0000;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px dashed #eee;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        try:
            st.image("University_of_Lahore_(logo).png", width=130)
        except:
            pass
        st.markdown("""
            <div style='font-size:18px; font-weight:600; margin-top:8px;'>University of Lahore</div>
            <div style='font-size:15px;'>Faculty of Information Technology</div>
            <div style='font-size:15px;'>Department of Computer Science</div>
            <div style='font-size:15px;'>Sargodha Campus</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        try:
            st.image("lab.jpeg", width=90)
        except:
            pass
        st.markdown("""
            <div style='font-size:17px; font-weight:600; margin-top:8px;'>Sponsored By</div>
            <div style='font-size:16px;'>VLCMatrix Lab</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin-top:25px;'>", unsafe_allow_html=True)
    
    # Load team images
    IMRAN_ASHRAF_IMG = get_img_as_base64("imranashraf.jpg")
    IMRAN_ASHRAF_SRC = f"data:image/jpeg;base64,{IMRAN_ASHRAF_IMG}" if IMRAN_ASHRAF_IMG else "https://via.placeholder.com/100?text=IA"
    
    SUPERVISOR_IMG = get_img_as_base64("supervisor.jpg")
    SUPERVISOR_SRC = f"data:image/jpg;base64,{SUPERVISOR_IMG}" if SUPERVISOR_IMG else "https://via.placeholder.com/100?text=S"
    
    FATIMA_IMG = get_img_as_base64("comitte head.jpg")
    FATIMA_SRC = f"data:image/jpeg;base64,{FATIMA_IMG}" if FATIMA_IMG else "https://via.placeholder.com/100?text=F"
    
    WASEEM_IMG = get_img_as_base64("hod.jpeg")
    WASEEM_SRC = f"data:image/jpeg;base64,{WASEEM_IMG}" if WASEEM_IMG else "https://via.placeholder.com/100?text=W"
    
    MOHSIN_IMG = get_img_as_base64("mohsin.jpeg")
    MOHSIN_SRC = f"data:image/jpeg;base64,{MOHSIN_IMG}" if MOHSIN_IMG else "https://via.placeholder.com/100?text=M"
    
    JUNAID_IMG = get_img_as_base64("jd.jpeg")
    JUNAID_SRC = f"data:image/jpeg;base64,{JUNAID_IMG}" if JUNAID_IMG else "https://via.placeholder.com/100?text=J"

    # Authorities Section
    st.markdown("<div class='section-title'>Concerned Authorities</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{SUPERVISOR_SRC}" alt="Dr. Tanzila Profile">
            <div class='team-title'>Dr. Tanzila Kehkashan</div>
            <div class='team-sub'>Assistant Professor</div>
            <div class='semester-info'>Supervisor</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{FATIMA_SRC}" alt="Ms. Fatima Iqbal Profile">
            <div class='team-title'>Ms. Fatima Iqbal</div>
            <div class='team-sub'>Lecturer</div>
            <div class='semester-info'>FYP Committee Head</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{WASEEM_SRC}" alt="Dr. Waseem Abbassi Profile">
            <div class='team-title'>Dr. Waseem Abbassi</div>
            <div class='team-sub'>Associate Professor</div>
            <div class='semester-info'>Head of Department (HoD)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Team Members Section
    st.markdown("<div class='section-title'>Project Team Members</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size: 1.1rem; color: #333;'>Session: Spring-22 to Fall-2025</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{IMRAN_ASHRAF_SRC}" alt="Imran Ashraf Profile">
            <div class='team-title'>Imran Ashraf</div>
            <div class='team-sub'>SAP ID: 70169493</div>
            <div class='semester-info'>Lead Researcher & System Developer</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{MOHSIN_SRC}" alt="Mohsin Profile">
            <div class='team-title'>Muhammad Mohsin</div>
            <div class='team-sub'>SAP ID: 70169511</div>
            <div class='semester-info'>Algorithm Researcher & Developer</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='team-card'>
            <img class='profile-img' src="{JUNAID_SRC}" alt="Junaid Profile">
            <div class='team-title'>Muhammad Junaid</div>
            <div class='team-sub'>SAP ID: 70159511</div>
            <div class='semester-info'>Documentation & UML Diagrams</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar Configuration
    st.sidebar.markdown("""
        ## 🛠️ Control Panel
        **Configure simulation and evaluation parameters.**
        ---
    """)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "🤖 MODEL EVALUATION",
        "📊 TRAINING METRICS", 
        "🔴 LIVE SIMULATION"
    ])
    
    # TAB 1: MODEL EVALUATION
    with tab1:
        st.markdown("<div class='section-title'>Trained Model Evaluation</div>", unsafe_allow_html=True)
        
        with st.spinner("Scanning for trained models..."):
            checkpoints = find_all_checkpoints()
        
        col1c, col2c = st.columns([2, 1])
        
        with col1c:
            st.markdown("### 📦 Available Models")
            mappo_count = len(checkpoints['MAPPO'])
            rmappo_count = len(checkpoints['RMAPPO'])
            
            st.success(f"**MAPPO:** {mappo_count} trained runs found")
            st.success(f"**RMAPPO:** {rmappo_count} trained runs found")
            
            with st.expander("📋 View All Runs"):
                st.markdown("**MAPPO Runs:**")
                for cp in checkpoints['MAPPO']:
                    st.text(f"  • {cp['run']} → {cp['path']}")
                
                st.markdown("**RMAPPO Runs:**")
                for cp in checkpoints['RMAPPO']:
                    st.text(f"  • {cp['run']} → {cp['path']}")
        
        with col2c:
            st.markdown("### 📊 Quick Stats")
            total_models = mappo_count + rmappo_count
            st.metric("Total Models", total_models)
            st.metric("Algorithms", "2")
        
        st.markdown("---")
        
        st.markdown("### 🚀 Run Model Evaluation")
        
        eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
        
        with eval_col1:
            algo_choice = st.selectbox(
                "Select Algorithm",
                options=["MAPPO", "RMAPPO"]
            )
        
        with eval_col2:
            available_runs = [cp['run'] for cp in checkpoints[algo_choice]]
            if available_runs:
                run_choice = st.selectbox(
                    "Select Run",
                    options=available_runs,
                    help=f"Choose from {len(available_runs)} available runs"
                )
            else:
                st.warning(f"No {algo_choice} runs found")
                run_choice = None
        
        with eval_col3:
            render_eval = st.checkbox(
                "Enable Rendering + Video",
                value=True,
                help="If checked, runs evaluation with rendering and records an evaluation video."
            )
        
        with eval_col4:
            num_agents_eval = st.number_input("Number of Agents", min_value=1, max_value=8, value=4)
        
        if run_choice:
            selected_checkpoint = next(
                (cp for cp in checkpoints[algo_choice] if cp['run'] == run_choice),
                None
            )
            
            if selected_checkpoint:
                st.info(f"📁 **Checkpoint Path:** `{selected_checkpoint['path']}`")
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("▶️ START EVALUATION", type="primary", use_container_width=True):
                        progress_placeholder = st.empty()
                        
                        if render_eval:
                            if not MSS_AVAILABLE:
                                st.error("The 'mss' package is required for recording. Install with: pip install mss")
                            else:
                                progress_placeholder.info(f"Running {algo_choice} - {run_choice} with video recording...")
                                
                                out_dir = EVALUATION_OUTPUT / f"{algo_choice}_{run_choice}"
                                try:
                                    video_path, stdout, stderr = evaluate_model_and_record_video(
                                        checkpoint_path=selected_checkpoint['path'],
                                        algo_name=selected_checkpoint['algo'],
                                        out_dir=out_dir,
                                        num_agents=num_agents_eval
                                    )
                                    progress_placeholder.success("✅ Evaluation completed and video recorded!")
                                    
                                    if video_path.exists():
                                        st.markdown("### 🎥 Evaluation Video")
                                        st.image(str(video_path))
                                    
                                    with st.expander("📄 Evaluation Output (stdout)"):
                                        st.code(stdout or "No output", language="text")
                                    if stderr:
                                        with st.expander("⚠️ Evaluation Errors (stderr)"):
                                            st.code(stderr, language="text")

                                except Exception as e:
                                    progress_placeholder.error(f"Evaluation with video failed: {e}")
                                    with st.expander("🔍 Error Details"):
                                        st.code(traceback.format_exc(), language="text")
                        else:
                            progress_placeholder.info(f"Evaluating {algo_choice} - {run_choice} (no video)...")
                            
                            result = evaluate_model_sync(
                                checkpoint_path=selected_checkpoint['path'],
                                algo_name=selected_checkpoint['algo'],
                                render=False,
                                num_agents=num_agents_eval
                            )
                            
                            if result['success']:
                                progress_placeholder.success("✅ Evaluation completed successfully!")

                                st.markdown("### 📈 Evaluation Results")

                                video_files = sorted(
                                    EVALUATION_OUTPUT.glob("**/*.gif"),
                                    key=lambda p: p.stat().st_mtime
                                )
                                if video_files:
                                    st.markdown("### 🎥 Evaluation Video")
                                    latest_video = video_files[-1]
                                    st.image(str(latest_video))
                                    st.info(f"Showing: {latest_video.name}")

                                with st.expander("📄 View Full Evaluation Output", expanded=True):
                                    st.code(result['stdout'], language='text')
                                
                                if result['stdout']:
                                    lines = result['stdout'].split('\n')
                                    metrics_found = False
                                    
                                    for line in lines:
                                        if 'reward' in line.lower() or 'success' in line.lower():
                                            st.text(line)
                                            metrics_found = True
                                    
                                    if not metrics_found:
                                        st.info("No specific metrics found in output. See full output above.")
                            else:
                                progress_placeholder.error("❌ Evaluation failed")
                                
                                st.error("**Error Details:**")
                                st.code(result['stderr'], language='text')
                                
                                if result['stdout']:
                                    with st.expander("📄 View Partial Output"):
                                        st.code(result['stdout'], language='text')
                                
                                st.warning("**Troubleshooting Tips:**")
                                st.markdown("""
                                - Ensure the checkpoint path contains valid model files (actor.pt, critic.pt)
                                - Check that the training script path is correct
                                - Verify Python environment has all required dependencies
                                """)
    
    # TAB 2: TRAINING METRICS
    with tab2:
        st.markdown("<div class='section-title'>Training Performance Analysis</div>", unsafe_allow_html=True)
        
        view_option = st.radio(
            "Select View:",
            ["📈 Algorithm Comparison (MAPPO vs RMAPPO)", "📊 Individual Algorithm Metrics"],
            horizontal=True
        )
        
        if view_option == "📈 Algorithm Comparison (MAPPO vs RMAPPO)":
            st.markdown("Compare training performance between MAPPO and RMAPPO algorithms")
            
            col1m, col2m = st.columns([3, 1])
            with col1m:
                comparison_type = st.selectbox(
                    "Select Comparison Type:",
                    ["Rewards Only", "All Metrics"]
                )
            
            if st.button("🔄 Generate Comparison", type="primary"):
                if comparison_type == "Rewards Only":
                    load_and_plot_mappo_rewards()
                else:
                    load_and_plot_all_metrics()
        
        else:
            st.markdown("View detailed metrics for a specific algorithm")
            
            algo_choice_metrics = st.selectbox(
                "Select Algorithm:",
                ["MAPPO", "RMAPPO"]
            )
            
            if st.button("🔄 Load Algorithm Metrics", type="primary"):
                load_individual_algorithm_metrics(algo_choice_metrics)
    
    # TAB 3: LIVE SIMULATION
    with tab3:
        st.markdown("<div class='section-title'>Live Environment Simulation</div>", unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        st.sidebar.header("🎮 Simulation Settings")
        
        scenario_options = {
            "X (Intersection)": "X",
            "O (Roundabout)": "O",
            "T (T-Junction)": "T",
            "CXSOT (Complex/Procedural)": "CXSOT",
            "S (Straight Road)": "S"
        }
        selected_scenario_label = st.sidebar.selectbox("Map Type", list(scenario_options.keys()))
        map_code = scenario_options[selected_scenario_label]
        
        seed = st.sidebar.number_input("Random Seed", value=42, step=1)
        num_agents = st.sidebar.slider("Controllable Agents", 1, 8, 4)
        
        st.sidebar.header("🌍 Environment")
        traffic_density = st.sidebar.slider("Traffic Density", 0.0, 1.0, 0.0, 0.05)
        traffic_mode = st.sidebar.selectbox("Traffic Behavior", ["hybrid", "respawn"])
        
        st.sidebar.header("📸 Capture")
        steps = st.sidebar.slider("Total Steps", 100, 1000, 300, 50)
        snaps = st.sidebar.slider("Snapshots", 1, 5, 3)
        
        scenario_dir_name = selected_scenario_label.split(" ")[0]
        out_dir = SCREENSHOT_ROOT / f"{scenario_dir_name}_{int(time.time())}"
        
        watermark_text = st.sidebar.text_input("Watermark", value=WATERMARK_TEXT_DEFAULT)
        
        st.markdown("### Current Configuration")
        
        config_col1, config_col2, config_col3, config_col4 = st.columns(4)
        config_col1.metric("Scenario", map_code)
        config_col2.metric("Agents", num_agents)
        config_col3.metric("Density", f"{traffic_density * 100:.0f}%")
        config_col4.metric("Seed", seed)
        
        config_col5, config_col6, config_col7, config_col8 = st.columns(4)
        config_col5.metric("Traffic Mode", traffic_mode)
        config_col6.metric("Steps", steps)
        config_col7.metric("Snapshots", snaps)
        config_col8.metric("Output", out_dir.name[:15] + "...")
        
        st.markdown("---")
        
        col_sim1, col_sim2, col_sim3 = st.columns([1, 2, 1])
        with col_sim2:
            if st.button("🚀 RUN SIMULATION & CAPTURE", type="primary", use_container_width=True):
                if not PANDA3D_AVAILABLE:
                    st.error("❌ MetaDrive environment unavailable. Check installation.")
                else:
                    status_placeholder = st.empty()
                    status_placeholder.info("🔄 Simulation starting... MetaDrive window will open.")
                    
                    try:
                        saved_files = run_simulation_and_capture(
                            map_code=map_code, 
                            num_agents=num_agents, 
                            traffic_density=traffic_density,
                            traffic_mode=traffic_mode, 
                            steps=steps, 
                            snaps=snaps, 
                            out_dir=out_dir,
                            watermark_text=watermark_text, 
                            seed=seed
                        )
                        
                        status_placeholder.success(f"✅ Simulation complete! Saved {len(saved_files)} files to `{out_dir}`")
                        
                        if saved_files:
                            st.markdown("### 📸 Captured Outputs")
                            for img_path in saved_files:
                                if img_path.lower().endswith(".gif"):
                                    st.image(img_path)
                                else:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"📷 {Path(img_path).name}", use_column_width=True)
                        
                    except Exception as e:
                        status_placeholder.error(f"❌ Simulation failed: {e}")
                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()