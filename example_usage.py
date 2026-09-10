import sys
from client import SGDRCosineScheduler

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def run():
    print(">>> Demonstrating SGDR Cosine Annealing with Restarts...")
    sched = SGDRCosineScheduler(base_lr=0.1, min_lr=0.001, t_0=5, t_mult=2)
    
    lrs = [sched.get_lr(ep) for ep in range(15)]
    print(f"Epoch 0 (Cycle 1 Start): LR = {lrs[0]:.4f}")
    print(f"Epoch 4 (Cycle 1 End):   LR = {lrs[4]:.4f}")
    print(f"Epoch 5 (Cycle 2 Reset): LR = {lrs[5]:.4f}")

    assert lrs[0] == 0.1
    assert lrs[4] < 0.05
    assert lrs[5] == 0.1 # Warm restart resets to base_lr
    print("[PASS] SGDR Cosine Annealing verified.")

if __name__ == "__main__":
    run()
