# KAGGLE DATASET INSPECTOR
# Run this first to understand the dataset structure

"""
INSTRUCTIONS:
1. Create a NEW Kaggle Notebook
2. Add the dataset: "land-mark-holistic-features-WLASL" or search for WLASL holistic
3. Copy this entire script
4. Run it
5. Copy the output and send it back
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

def inspect_dataset():
    """Comprehensive dataset structure inspection"""
    
    print("=" * 80)
    print("📂 KAGGLE DATASET STRUCTURE INSPECTOR")
    print("=" * 80)
    
    base_path = Path("/kaggle/input")
    
    print(f"\n1️⃣ Available Datasets in /kaggle/input:\n")
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            print(f"   📁 {folder.name}")
    
    # Find WLASL-related folders
    wlasl_folders = [f for f in base_path.iterdir() 
                     if f.is_dir() and any(term in f.name.lower() 
                     for term in ['wlasl', 'holistic', 'landmark', 'sign'])]
    
    if not wlasl_folders:
        print("\n⚠️ No WLASL/holistic datasets found!")
        return
    
    for data_path in wlasl_folders:
        print(f"\n" + "=" * 80)
        print(f"2️⃣ Inspecting: {data_path.name}")
        print("=" * 80)
        
        # Count files by type
        all_files = list(data_path.rglob("*"))
        files_only = [f for f in all_files if f.is_file()]
        dirs_only = [f for f in all_files if f.is_dir()]
        
        print(f"\n📊 File Statistics:")
        print(f"   Total files: {len(files_only)}")
        print(f"   Total directories: {len(dirs_only)}")
        
        # Group by extension
        extensions = {}
        for f in files_only:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print(f"\n📝 Files by type:")
        for ext, count in sorted(extensions.items(), key=lambda x: -x[1]):
            ext_name = ext if ext else "(no extension)"
            print(f"   {ext_name}: {count}")
        
        # Show directory structure (first 2 levels)
        print(f"\n🌳 Directory Structure (top 2 levels):")
        shown_dirs = set()
        for item in sorted(data_path.rglob("*"))[:200]:  # Limit to first 200
            rel_path = item.relative_to(data_path)
            parts = rel_path.parts
            if len(parts) <= 2:
                if len(parts) == 1:
                    prefix = "   ├── "
                else:
                    prefix = "   │   ├── "
                
                display_path = str(rel_path)
                if display_path not in shown_dirs:
                    filetype = "📁" if item.is_dir() else "📄"
                    size_info = ""
                    if item.is_file() and item.stat().st_size < 1e9:  # < 1GB
                        size_mb = item.stat().st_size / (1024 * 1024)
                        size_info = f" ({size_mb:.1f} MB)"
                    print(f"{prefix}{filetype} {display_path}{size_info}")
                    shown_dirs.add(display_path)
        
        # Check for NPY files specifically
        npy_files = list(data_path.rglob("*.npy"))
        print(f"\n🔍 NPY File Analysis:")
        print(f"   Total NPY files: {len(npy_files)}")
        
        if npy_files:
            print(f"\n   Sample NPY files (first 10):")
            for npy in npy_files[:10]:
                rel = npy.relative_to(data_path)
                print(f"      • {rel}")
                
                # Try to load and check shape
                try:
                    data = np.load(npy, allow_pickle=True)
                    if isinstance(data, np.ndarray):
                        print(f"        Shape: {data.shape}, Dtype: {data.dtype}")
                        if len(data.shape) == 1 and data.shape[0] < 100:
                            # Might be an array of objects
                            print(f"        First element type: {type(data[0]) if len(data) > 0 else 'empty'}")
                    else:
                        print(f"        Type: {type(data)}")
                except Exception as e:
                    print(f"        ⚠️ Error loading: {e}")
        
        # Check for CSV/Parquet
        csv_files = list(data_path.rglob("*.csv"))
        parquet_files = list(data_path.rglob("*.parquet"))
        
        if csv_files:
            print(f"\n📊 CSV File Analysis:")
            print(f"   Total CSV files: {len(csv_files)}")
            print(f"   Sample (first 3):")
            for csv in csv_files[:3]:
                rel = csv.relative_to(data_path)
                print(f"      • {rel}")
                try:
                    df = pd.read_csv(csv, nrows=5)
                    print(f"        Columns: {list(df.columns)[:10]}")
                    print(f"        Shape: {df.shape}")
                except Exception as e:
                    print(f"        ⚠️ Error: {e}")
        
        if parquet_files:
            print(f"\n📊 Parquet File Analysis:")
            print(f"   Total Parquet files: {len(parquet_files)}")
            print(f"   Sample (first 3):")
            for pq in parquet_files[:3]:
                rel = pq.relative_to(data_path)
                print(f"      • {rel}")
                try:
                    df = pd.read_parquet(pq)
                    print(f"        Columns: {list(df.columns)[:10]}")
                    print(f"        Shape: {df.shape}")
                except Exception as e:
                    print(f"        ⚠️ Error: {e}")
        
        # Check for JSON
        json_files = list(data_path.rglob("*.json"))
        if json_files and len(json_files) < 20:
            print(f"\n📋 JSON files found: {len(json_files)}")
            for jf in json_files[:5]:
                print(f"      • {jf.relative_to(data_path)}")

if __name__ == "__main__":
    inspect_dataset()
    print("\n" + "=" * 80)
    print("✅ Inspection complete! Copy this output and send it back.")
    print("=" * 80)
