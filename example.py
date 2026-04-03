# Detect environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab - Perfect!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Nice!")

if IN_COLAB:
    import subprocess
    import os
    print("\n📦 Cloning OpenEnv repository...")
    subprocess.run(["git", "clone", "https://github.com/meta-pytorch/OpenEnv.git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chdir("OpenEnv")
    
    print("📚 Installing dependencies (this takes ~10 seconds)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "requests"])
    
    import sys
    sys.path.insert(0, './src')
    print("\n✅ Setup complete! Everything is ready to go! 🎉")
else:
    import sys
    from pathlib import Path
    # Corrected: Point to the current folder so 'import src.xxx' works
    sys.path.insert(0, str(Path.cwd()))
    print("✅ Using local OpenEnv installation")

# Testing the connection
try:
    from src.environment import SupportEnv
    from src.models import TaskName
    env = SupportEnv(task=TaskName.EASY)
    print(f"🚀 Environment '{env.task_name.value}' initialized successfully!")
except Exception as e:
    print(f"❌ Setup error: {e}")

print("\n🚀 Ready to explore OpenEnv and build amazing things!")
print("💡 Tip: Run cells top-to-bottom for the best experience.\n")