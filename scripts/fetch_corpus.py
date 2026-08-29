"""Fetch YouTube thumbnails into tests/corpus/ for detection triage."""
import sys
import urllib.request
from pathlib import Path

IDS = [
    "VwTBkTuzBGw", "cVST-605PfY", "bcBFPmfntyo", "jR3rWCBeO6M",
    "HOEBZy0NB68", "D57Bum-y_Bk", "1wLNzcrGrVw", "ajAXAI6RPOE",
    "AgiR798L_bQ", "9sHysLh0b2Q", "m-EyrzSNBII", "8IzgsqasAmw",
]
out = Path("tests/corpus")
out.mkdir(parents=True, exist_ok=True)
ok = 0
for vid in IDS:
    dest = out / f"yt_{vid}.jpg"
    if dest.exists():
        ok += 1
        continue
    for quality in ("maxresdefault", "hqdefault"):
        url = f"https://i.ytimg.com/vi/{vid}/{quality}.jpg"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            if len(data) > 3000:
                dest.write_bytes(data)
                ok += 1
                print(f"{vid}: {quality} {len(data)//1024}KB")
                break
        except Exception as e:
            print(f"{vid}: {quality} failed {e}")
print(f"fetched {ok}/{len(IDS)}")
