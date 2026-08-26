"""Direct smart run with full traceback."""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.jobs import _load_upload
import pipeline

img = _load_upload("tests/samples/before.jpeg")
t0 = time.time()
try:
    meta = pipeline.run(img, ratio="9:16", tier="smart", job_id="smart-direct-2")
    print(f"WALL {time.time()-t0:.0f}s | engine={meta['engine']} | "
          f"cards={meta.get('cards_composed')} | fidelity={meta.get('fidelity_ok')}")
    print("void:", meta["void"])
except Exception:
    traceback.print_exc()
