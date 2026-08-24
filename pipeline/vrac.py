"""VRAM discipline: exactly one model resident at a time, guaranteed cleanup."""
import gc
import logging
from contextlib import contextmanager

log = logging.getLogger("h2v.vrac")

_state = {"resident": None}


def resident() -> str | None:
    return _state["resident"]


@contextmanager
def model_slot(name: str, loader=None, unloader=None):
    """Exclusive residency guard.

    with model_slot("lama", load_lama, unload_lama) as lama:
        ...

    Raises RuntimeError if another model is still resident - that is a bug in
    orchestration and must fail loudly, never silently swap models on GPU.
    """
    if _state["resident"] is not None:
        raise RuntimeError(f"VRAM slot busy: '{_state['resident']}' still resident "
                           f"(tried to load '{name}')")
    from config import config  # config already set TORCH_HOME pre-torch
    _state["resident"] = name
    handle = loader() if loader else None
    try:
        yield handle
    finally:
        if unloader and handle is not None:
            try:
                unloader(handle)
            except Exception as e:
                log.warning("unloader for %s raised: %s", name, e)
        del handle
        _empty_cache()
        _state["resident"] = None


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def release_all():
    """Belt-and-braces reset used by worker between jobs."""
    _empty_cache()
