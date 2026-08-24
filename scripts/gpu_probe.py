import subprocess
import time

import torch

a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    b = a @ a
torch.cuda.synchronize()
print("matmul ms:", round((time.time() - t0) / 10 * 1000), flush=True)
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=clocks.gr,power.draw,clocks_throttle_reasons.active,temperature.gpu",
     "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
