
#!/usr/bin/env python3
"""
Install sentence-transformers and PyTorch with M1 GPU support
"""

import subprocess
import sys

def install_gpu_packages():
    print("\n" + "="*60)
    print("Installing Local Embedding Dependencies")
    print("="*60 + "\n")
    
    packages = [
        ("torch", "PyTorch with M1 GPU support"),
        ("sentence-transformers", "Sentence transformers library"),
    ]
    
    for package, description in packages:
        print(f"\nInstalling: {package}")
        print(f"Purpose: {description}")
        print("-"*60)
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    print("\n" + "="*60)
    print("Testing M1 GPU (MPS) Support")
    print("="*60 + "\n")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("✓ M1/M2 GPU (MPS) is AVAILABLE!")
            print("✓ Your embeddings will be MUCH faster!")
            
            # Test GPU
            x = torch.ones(1, device="mps")
            print("✓ Successfully created tensor on GPU")
        else:
            print("⚠ M1 GPU not available, will use CPU")
            print("  (This is still faster than OpenAI API)")
    except Exception as e:
        print(f"✗ Error testing GPU: {e}")
    
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nNext step:")
    print("Update your .env file:")
    print("  EMBEDDING_TYPE=local")
    print("  USE_GPU=true")
    print("\nThen run: python data_preprocessing.py")
    print("\n")
    
    return True

if __name__ == "__main__":
    install_gpu_packages()