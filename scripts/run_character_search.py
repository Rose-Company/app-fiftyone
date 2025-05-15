#!/usr/bin/env python3
import subprocess
import os
import sys
import time

# Check if docker environment (presence of /app)
def is_docker_environment():
    return os.path.exists("/app")

def run_script(script_path):
    """Run a Python script and display its output in real-time"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_path}")
    print(f"{'='*80}")
    
    # Determine the correct path based on environment
    if is_docker_environment():
        full_path = f"/app/scripts/{script_path}"
        cmd = ["python", full_path]
    else:
        # For local development, use the relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, script_path)
        cmd = ["python", full_path]
    
    print(f"Executing: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the command
    process = subprocess.Popen(
        cmd,
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
        print(f"\nError: Command failed with exit code {exit_code}")
        return False
    
    print("\nCommand completed successfully.")
    return True

def main():
    print("FiftyOne Character Search Utility")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if all required datasets exist
    success = run_script("character_search.py")
    
    if success:
        print("\nCharacter search system is ready!")
        print("You can access the interface at http://localhost:5151")
    else:
        print("\nCharacter search setup failed.")
        print("Make sure you have completed all previous pipeline steps:")
        print("1. Import videos")
        print("2. Extract frames")
        print("3. Detect people")
        print("4. Extract faces")
        print("5. Group faces into characters")

if __name__ == "__main__":
    main() 