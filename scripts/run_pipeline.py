#!/usr/bin/env python3
import subprocess
import time
import os
import sys

def run_command(command, description):
    """
    Run a shell command and display its output in real-time
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(command)}")
    print("-" * 80)
    
    # Run the command
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    # Wait for process to complete
    exit_code = process.wait()
    
    if exit_code != 0:
        print(f"\nERROR: Command failed with exit code {exit_code}")
        print("Stopping pipeline execution.")
        return False
    
    print("\nCommand completed successfully.")
    return True

def main():
    # List of steps in our pipeline
    steps = [
        {
            "command": ["python", "/app/scripts/import_videos.py"],
            "description": "Import videos into FiftyOne dataset"
        },
        {
            "command": ["python", "/app/scripts/extract_frames.py"],
            "description": "Extract frames from videos"
        },
        {
            "command": ["python", "/app/scripts/detect_people.py"],
            "description": "Detect people in frames"
        },
        {
            "command": ["python", "/app/scripts/extract_faces.py"],
            "description": "Extract faces from frames with people"
        },
        {
            "command": ["python", "/app/scripts/group_faces_characters.py"],
            "description": "Group faces into characters"
        },
        {
            "command": ["python", "/app/scripts/character_search.py"],
            "description": "Create character index and launch search interface"
        }
    ]
    
    print("Starting FiftyOne Video Analysis Pipeline")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run each step in sequence
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}/{len(steps)}: {step['description']}")
        
        success = run_command(step["command"], step["description"])
        if not success:
            break
        
        # Small delay between steps to allow resources to be freed
        if i < len(steps):
            print("Waiting for resources to be freed...")
            time.sleep(5)
    
    print("\nPipeline execution completed.")
    print("You can view the results at http://localhost:5151")

if __name__ == "__main__":
    main() 