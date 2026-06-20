"""
Simple MetaDrive Evaluation with Video Recording
This script tests if MetaDrive rendering works and records a video
"""

import os
import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

def test_metadrive_rendering():
    """
    Test if MetaDrive can render and save video
    """
    print("=" * 70)
    print("TESTING METADRIVE RENDERING & VIDEO RECORDING")
    print("=" * 70)
    
    output_dir = "FYP_SUBMISSION"
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Try with visible window
    print("\n[TEST 1] Trying with visible window...")
    try:
        env = MetaDriveEnv(config={
            "use_render": True,
            "manual_control": False,
            "num_scenarios": 1,
        })
        
        print("✓ Environment created successfully!")
        print("  If a window opens, your rendering works!")
        print("  You can use screen recording software to record it.")
        
        obs, info = env.reset()
        
        # Run for a few steps
        for i in range(50):
            obs, reward, terminated, truncated, info = env.step([0, 1])
            if terminated or truncated:
                break
        
        env.close()
        print("✓ Test 1 passed - visible window works!")
        return True
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        print("  Trying offscreen rendering...")
    
    # Method 2: Try offscreen rendering with rgb_array
    print("\n[TEST 2] Trying offscreen rendering with frame capture...")
    try:
        env = MetaDriveEnv(config={
            "use_render": False,
            "offscreen_render": True,
            "manual_control": False,
            "num_scenarios": 1,
        })
        
        print("✓ Offscreen environment created!")
        
        obs, info = env.reset()
        
        # Try to get a frame
        try:
            frame = env.render(mode='rgb_array')
            print(f"✓ Frame captured! Shape: {frame.shape}")
            
            # Save sample frame
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/sample_frame.png")
            print(f"✓ Sample frame saved: {output_dir}/sample_frame.png")
            
            env.close()
            return True
            
        except Exception as e2:
            print(f"✗ Could not render frame: {e2}")
            env.close()
            
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Method 3: Try top-down rendering (most reliable)
    print("\n[TEST 3] Trying top-down rendering...")
    try:
        env = MetaDriveEnv(config={
            "use_render": False,
            "offscreen_render": False,
            "manual_control": False,
            "num_scenarios": 1,
        })
        
        print("✓ Environment created for top-down rendering!")
        
        obs, info = env.reset()
        
        # Try top-down render
        try:
            frame = env.render(mode='topdown', film_size=(800, 800))
            print(f"✓ Top-down frame captured! Shape: {frame.shape}")
            
            # Save frame
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/topdown_frame.png")
            print(f"✓ Top-down frame saved: {output_dir}/topdown_frame.png")
            
            env.close()
            return True
            
        except Exception as e3:
            print(f"✗ Could not render top-down: {e3}")
            env.close()
            
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    return False


def record_metadrive_video(num_steps=300, use_topdown=False):
    """
    Record a video of MetaDrive environment
    """
    print("\n" + "=" * 70)
    print("RECORDING METADRIVE VIDEO")
    print("=" * 70)
    
    output_dir = "FYP_SUBMISSION"
    os.makedirs(output_dir, exist_ok=True)
    
    render_mode = "topdown" if use_topdown else "rgb_array"
    output_file = f"{output_dir}/metadrive_{'topdown' if use_topdown else 'demo'}.mp4"
    
    print(f"\nRender mode: {render_mode}")
    print(f"Output file: {output_file}")
    print(f"Recording {num_steps} steps...")
    
    try:
        # Create environment
        env_config = {
            "use_render": False,
            "offscreen_render": not use_topdown,
            "manual_control": False,
            "num_scenarios": 1,
            "start_seed": 1000,
        }
        
        env = MetaDriveEnv(config=env_config)
        obs, info = env.reset()
        
        print("✓ Environment created")
        
        # Get first frame to determine size
        if use_topdown:
            first_frame = env.render(mode='topdown', film_size=(800, 800))
        else:
            first_frame = env.render(mode='rgb_array')
        
        height, width = first_frame.shape[:2]
        print(f"✓ Frame size: {width}x{height}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("✗ Failed to create video writer")
            print("Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_file = output_file.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Could not create video writer with any codec")
        
        print("✓ Video writer initialized")
        print("\n[INFO] Recording frames...")
        
        frames_written = 0
        
        for step in range(num_steps):
            # Take a step
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Render frame
            if use_topdown:
                frame = env.render(mode='topdown', film_size=(800, 800))
            else:
                frame = env.render(mode='rgb_array')
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(frame_bgr)
            frames_written += 1
            
            if (step + 1) % 50 == 0:
                print(f"  Progress: {step + 1}/{num_steps} frames")
            
            # Reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()
        
        # Cleanup
        out.release()
        env.close()
        
        print(f"\n✓ Video saved successfully!")
        print(f"✓ Output: {output_file}")
        print(f"✓ Frames: {frames_written}")
        print(f"✓ Duration: {frames_written/fps:.1f} seconds")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error recording video: {e}")
        print("\nDEBUGGING INFO:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\nMETADRIVE VIDEO RECORDING FOR FYP")
    print("=" * 70)
    
    # First, test if rendering works
    print("\nStep 1: Testing MetaDrive rendering capabilities...")
    rendering_works = test_metadrive_rendering()
    
    if not rendering_works:
        print("\n" + "=" * 70)
        print("⚠ RENDERING TESTS FAILED")
        print("=" * 70)
        print("\nPossible solutions:")
        print("1. Install required packages:")
        print("   pip install panda3d pillow opencv-python")
        print("\n2. On Windows, you may need:")
        print("   pip install pygame")
        print("\n3. For GPU rendering issues:")
        print("   Update your graphics drivers")
        print("\n4. Use manual screen recording instead:")
        print("   - Run: python -m metadrive.examples.drive_in_single_agent_env")
        print("   - Record with Windows Game Bar (Win + G)")
        return
    
    print("\n✓ Rendering works!")
    
    # Now try to record video
    print("\nStep 2: Recording demonstration video...")
    
    # Try top-down first (most reliable)
    print("\nAttempting top-down view recording...")
    video1 = record_metadrive_video(num_steps=200, use_topdown=True)
    
    # Try 3D view
    print("\nAttempting 3D view recording...")
    video2 = record_metadrive_video(num_steps=200, use_topdown=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if video1:
        print(f"✓ Top-down video: {video1}")
    if video2:
        print(f"✓ 3D view video: {video2}")
    
    if video1 or video2:
        print("\n✓ SUCCESS! You have video(s) for your FYP submission!")
        print("\nNext steps:")
        print("1. Check the FYP_SUBMISSION folder")
        print("2. Include the video in your report")
        print("3. Show it to your supervisor")
    else:
        print("\n⚠ Automated recording failed")
        print("\nUse manual screen recording:")
        print("1. Run: python -m metadrive.examples.drive_in_single_agent_env")
        print("2. Press Win + G to record")
        print("3. Save the recording")


if __name__ == "__main__":
    main()