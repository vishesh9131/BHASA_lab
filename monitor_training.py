#!/usr/bin/env python3
"""
BHASA Mamba Training Monitor
Monitor the training progress and display current status
"""

import os
import time
import torch
import json
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check current training status"""
    print("🔍 BHASA Mamba Training Monitor")
    print("=" * 50)
    
    # Check if training process is running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'bhasa_mamba.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process is running")
            print(f"📊 Process ID: {result.stdout.strip()}")
        else:
            print("❌ No training process found")
    except:
        print("⚠️ Could not check process status")
    
    # Check checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        print(f"\n💾 Found {len(checkpoints)} checkpoint files:")
        
        for cp in sorted(checkpoints):
            size_mb = cp.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(cp.stat().st_mtime)
            print(f"   📁 {cp.name} ({size_mb:.1f} MB) - {mod_time.strftime('%H:%M:%S')}")
        
        # Try to load latest checkpoint info
        latest_cp = checkpoint_dir / "bhasa_latest.pt"
        if latest_cp.exists():
            try:
                checkpoint = torch.load(latest_cp, map_location='cpu')
                epoch = checkpoint.get('epoch', 'Unknown')
                loss = checkpoint.get('loss', 'Unknown')
                print(f"\n📈 Latest checkpoint:")
                print(f"   🎯 Epoch: {epoch}")
                print(f"   📉 Loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"   📉 Loss: {loss}")
            except Exception as e:
                print(f"\n⚠️ Could not read checkpoint: {e}")
    else:
        print("\n❌ No checkpoints directory found")
    
    # Check tokenizer
    tokenizer_file = Path("bhasa_tokenizer.json")
    if tokenizer_file.exists():
        try:
            with open(tokenizer_file, 'r') as f:
                tokenizer_data = json.load(f)
            vocab_size = tokenizer_data.get('vocab_size', 'Unknown')
            print(f"\n🔤 Tokenizer: {vocab_size} vocabulary size")
        except:
            print("\n⚠️ Could not read tokenizer")
    
    # Check final model
    final_model = Path("bhasa_mamba_final.pt")
    if final_model.exists():
        size_mb = final_model.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(final_model.stat().st_mtime)
        print(f"\n🎉 Final model exists: {size_mb:.1f} MB ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    print("\n" + "=" * 50)

def monitor_loop():
    """Continuous monitoring loop"""
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            check_training_status()
            print(f"\n⏰ Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print("🔄 Refreshing in 30 seconds...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        monitor_loop()
    else:
        check_training_status()
        print("\n💡 Use --loop for continuous monitoring") 