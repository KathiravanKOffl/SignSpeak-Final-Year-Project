#!/usr/bin/env python3
"""
Create label_mapping_123.json from file_to_label.json (training output)

Usage:
    python create_label_mapping.py /path/to/file_to_label.json

Output:
    backend/checkpoints/label_mapping_123.json
"""

import json
import sys
from pathlib import Path

def create_label_mapping(file_to_label_path: str):
    """Convert file_to_label.json to label_mapping_123.json"""
    
    # Load file_to_label.json
    print(f"📖 Loading {file_to_label_path}...")
    with open(file_to_label_path, 'r') as f:
        file_to_label = json.load(f)
    
    print(f"✅ Loaded {len(file_to_label)} samples")
    
    # Extract unique classes (sorted alphabetically)
    unique_classes = sorted(set(file_to_label.values()))
    print(f"✅ Found {len(unique_classes)} unique classes")
    
    # Create index → class_name mapping
    label_mapping = {str(i): class_name for i, class_name in enumerate(unique_classes)}
    
    # Save to backend/checkpoints/
    output_path = Path('backend/checkpoints/label_mapping_123.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n✅ Created: {output_path}")
    print(f"📊 Classes: {len(label_mapping)}")
    print(f"\nFirst 10 classes:")
    for i in range(min(10, len(label_mapping))):
        print(f"  {i}: {label_mapping[str(i)]}")
    
    print(f"\nLast 3 classes:")
    for i in range(max(0, len(label_mapping) - 3), len(label_mapping)):
        print(f"  {i}: {label_mapping[str(i)]}")
    
    return label_mapping

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("❌ Error: Please provide path to file_to_label.json")
        print(f"\nUsage: python {sys.argv[0]} /path/to/file_to_label.json")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    create_label_mapping(file_path)
    print("\n🎉 Done! Now you can deploy the backend.")
