"""Subject detection: YOLO person/object passes -> saliency fallback -> None."""
import logging
import numpy as np

from config import config

log = logging.getLogger("h2v.detect")


def _area_weighted_centroid_y(results, img_h: int):
    """Area-weighted mean normalized y over detection boxes/masks. None if empty."""
    total_w, acc = 0.0, 0.0
    for r in results:
        if getattr(r, "masks", None) is not None and r.masks is not None \
                and getattr(r.masks, "data", None) is not None and len(r.masks.data):
            for m in r.masks.data.cpu().numpy():
                ys, _ = np.nonzero(m)
                if len(ys) == 0:
                    continue
                area = len(ys)
                acc += float(ys.mean()) / img_h * area
                total_w += area
        elif getattr(r, "boxes", None) is not None and len(r.boxes):
            for b in r.boxes.xyxy.cpu().numpy():
                bw = max(1.0, float(b[2] - b[0]))
                bh = max(1.0, float(b[3] - b[1]))
                acc += ((b[1] + b[3]) / 2.0 / img_h) * bw * bh
                total_w += bw * bh
    return (acc / total_w) if total_w > 0 else None


def _saliency_centroid_y(img_rgb: np.ndarray):
    import cv2
    gray = np.ascontiguousarray(
        np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8))
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, m = sal.computeSaliency(gray)
        energy = m.astype(np.float64) if ok else None
    except Exception:
        energy = None
    if energy is None:  # headless builds without saliency contrib -> gradient energy
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        energy = np.abs(gx) + np.abs(gy)
        energy -= energy.min()
        d = energy.max()
        energy = (energy / d).astype(np.float64) if d > 0 else None
    if energy is None or energy.sum() <= 0:
        return None
    s = energy / energy.sum()
    ys = np.arange(s.shape[0], dtype=np.float64)
    return float((s.sum(axis=1) * ys).sum()) / s.shape[0]


def subject_centroid_y(img_rgb: np.ndarray):
    """Return (centroid_norm|None, method:str). Never raises."""
    h, w = img_rgb.shape[:2]
    dev = 0 if config.device() == "cuda" else "cpu"
    try:
        from pipeline.vrac import model_slot
        from ultralytics import YOLO

        def _load():
            return YOLO(str(config.YOLO_PT_PATH))

        with model_slot("yolo", _load, None) as model:
            r1 = model.predict(img_rgb, conf=config.YOLO_PERSON_CONF,
                               classes=[0], device=dev, verbose=False)
            c = _area_weighted_centroid_y(r1, h)
            if c is not None:
                return c, "yolo_person"
            r2 = model.predict(img_rgb, conf=config.YOLO_ANY_CONF,
                               device=dev, verbose=False)
            c = _area_weighted_centroid_y(r2, h)
            if c is not None:
                return c, "yolo_any"
    except Exception as e:
        log.warning("YOLO detection failed (%s); falling back to saliency", e)

    c = _saliency_centroid_y(img_rgb)
    if c is not None:
        return c, "saliency"
    return None, "none"


if __name__ == "__main__":
    import sys
    import time
    from PIL import Image
    t0 = time.time()
    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    c, method = subject_centroid_y(img)
    print(f"centroid={c} method={method} elapsed={time.time()-t0:.2f}s")
