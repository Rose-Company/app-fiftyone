#!/usr/bin/env python3
"""
Test script to verify all Python scripts can read environment variables correctly
"""

import os
import sys
import subprocess
import importlib.util

# List of scripts to test
SCRIPTS_TO_TEST = [
    'import_videos.py',
    'extract_frames.py', 
    'extract_faces_deepface.py',
    'character_search.py',
    'remove_frames_by_video_id.py',
    'group_faces_graph.py',
    'face_recognition_dic.py',
    'face_recognition_utils.py',
    'remove_duplicate_frames.py'
]

# Required environment variables
REQUIRED_ENV_VARS = [
    'FIFTYONE_DATABASE_URI',
    'FIFTYONE_PORT',
    'VIDEO_DIR'
]

def test_env_vars():
    """Test if all required environment variables are set"""
    print("=== Testing Environment Variables ===")
    missing_vars = []
    
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("\n✅ All required environment variables are set!")
        return True

def test_script_imports():
    """Test if scripts can be imported without errors"""
    print("\n=== Testing Script Imports ===")
    results = {}
    
    for script in SCRIPTS_TO_TEST:
        script_path = f"/app/scripts/{script}"
        print(f"Testing {script}...")
        
        try:
            # Try to load the script
            spec = importlib.util.spec_from_file_location("test_module", script_path)
            if spec is None:
                results[script] = "❌ Cannot load script"
                continue
                
            module = importlib.util.module_from_spec(spec)
            
            # Try to execute the script (this will test environment variable loading)
            try:
                spec.loader.exec_module(module)
                results[script] = "✅ Environment variables loaded successfully"
            except SystemExit:
                results[script] = "✅ Script executed (exited normally)"
            except Exception as e:
                # Check if it's an environment variable error
                if "environment variable" in str(e).lower():
                    results[script] = f"❌ Environment variable error: {e}"
                else:
                    results[script] = f"⚠️  Script error (not env related): {type(e).__name__}"
                    
        except Exception as e:
            results[script] = f"❌ Import error: {e}"
    
    # Print results
    print("\n=== Test Results ===")
    for script, result in results.items():
        print(f"{script}: {result}")
    
    return results

def test_database_connection():
    """Test database connection"""
    print("\n=== Testing Database Connection ===")
    try:
        import fiftyone as fo
        
        # Set database URI from environment
        database_uri = os.getenv("FIFTYONE_DATABASE_URI")
        if not database_uri:
            print("❌ FIFTYONE_DATABASE_URI not set")
            return False
            
        fo.config.database_uri = database_uri
        
        # Try to list datasets
        datasets = fo.list_datasets()
        print(f"✅ Database connection successful! Found {len(datasets)} datasets:")
        for ds_name in datasets[:5]:  # Show first 5 datasets
            print(f"  - {ds_name}")
        if len(datasets) > 5:
            print(f"  ... and {len(datasets) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 FiftyOne Environment Variables Test Suite")
    print("=" * 50)
    
    # Test 1: Environment variables
    env_ok = test_env_vars()
    
    # Test 2: Database connection
    db_ok = test_database_connection()
    
    # Test 3: Script imports (only if env vars are ok)
    if env_ok:
        script_results = test_script_imports()
        
        # Count successful tests
        success_count = sum(1 for result in script_results.values() 
                          if result.startswith("✅"))
        total_count = len(script_results)
        
        print(f"\n=== Final Summary ===")
        print(f"Environment Variables: {'✅' if env_ok else '❌'}")
        print(f"Database Connection: {'✅' if db_ok else '❌'}")
        print(f"Script Tests: {success_count}/{total_count} passed")
        
        if success_count == total_count and env_ok and db_ok:
            print("\n🎉 All tests passed! Environment is properly configured.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please check the configuration.")
            return 1
    else:
        print("\n❌ Environment variables not properly set. Skipping script tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 