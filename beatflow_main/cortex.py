"""Adapter for the cortex-sama model (``Sora`` branch).

The model repo is **not** vendored here. Point ``model_dir`` at a checkout of
https://github.com/Sora3301/cortex-sama (``Sora`` branch); ``torch`` is imported
lazily so BeatFlow still runs on machines that cannot install it.

Contract this adapter implements (must match training):

* input is RGB, 320x180 (W x H), ``float / 255``, tensor ``(1, 1, 3, 180, 320)``;
* the model emits, per frame, one type per slot (idle / down / hold / flick)
  and one normalised coordinate per slot in ``[-1, 1]``;
* the coordinate space is the label space, so a contact maps 1:1 to
  :class:`beatflow_input.RawContact` — no pixel conversion, no id.

Streaming state is two things, not one
--------------------------------------
``Model.forward(x, h=h)`` carries the GRU hidden state, but the coordinate
branch also has a causal TCN whose cache is **not returned** by
``Model._forward`` (``local, _ = self.tcn(...)``), and ``forward(x, h=...)``
never passes it. Per-frame ``T=1`` inference therefore sees zero-padded
history instead of the 16 warmup frames training always has, and coordinates
degrade.

Until the model exposes the cache, :meth:`CortexModel._patch_tcn` wraps
``CausalTCN.forward`` so the newest cache is kept on the module and fed back on
the next call. Delete that patch once ``_forward`` returns the cache.

Usage::

    from beatflow_main.cortex import CortexModel

    model = CortexModel(
        model_dir="../cortex-sama",
        ckpt="../cortex-sama/checkpoints/1113/epoch_011.pt",
        device="cuda",
        min_score=0.0,
    )
    loop = AgentLoop(source, model, injector, stream, fps=60.0)
    loop.run()

``Frame.image`` may be ``bytes`` (RGB24), a ``(H, W, 3)`` uint8 ndarray, or a
``torch.Tensor``. Anything that is not already 320x180 is resized with bilinear
interpolation, so a full-resolution capture works too; keep the aspect ratio the
same as the training videos.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from beatflow_input import RawContact

from .frame_source import Frame
from .model import contact_or_none

__all__ = [
    "CortexModel",
    "contacts_from_types",
    "torch_available",
    "TYPE_IDLE",
    "TYPE_DOWN",
    "TYPE_HOLD",
    "TYPE_FLICK",
]

#: ``config.Action_type`` of cortex-sama, mirrored here so this module does not
#: need the model repo's constants to be importable at import time.
TYPE_IDLE = 0
TYPE_DOWN = 1
TYPE_HOLD = 2
TYPE_FLICK = 3

_KIND = {
    TYPE_DOWN: "down",
    TYPE_HOLD: "hold",
    TYPE_FLICK: "flick",
}


def torch_available() -> bool:
    """Whether the current interpreter can import torch."""
    return importlib.util.find_spec("torch") is not None


def _import_torch():
    if not torch_available():
        raise RuntimeError(
            "the cortex adapter needs PyTorch, which is not importable from "
            f"{sys.executable}. Run the main program in a 3.12/3.11 venv with "
            "torch installed (`python3.12 -m venv .venv`) - BeatFlow itself "
            "stays stdlib-only."
        )
    import torch  # noqa: PLC0415 - deliberately lazy

    return torch


def contacts_from_types(
    types: Sequence[int],
    coords: Sequence[Sequence[float]],
    probs: Sequence[Sequence[float]] | None = None,
    min_score: float = 0.0,
) -> tuple[RawContact, ...]:
    """One frame of ``(types, coords)`` -> contacts, dropping idle slots.

    ``types`` is the argmax per slot and ``coords`` the matching ``(x, y)``.
    ``probs`` is optional but required for ``min_score`` to do anything: a
    prediction below the threshold is dropped, which trades a missed note for
    fewer phantom taps.
    """
    out: list[RawContact] = []
    for slot, raw_type in enumerate(types):
        type_id = int(raw_type)
        kind = _KIND.get(type_id)
        if kind is None:
            continue
        if min_score > 0.0 and probs is not None:
            if float(probs[slot][type_id]) < min_score:
                continue
        x, y = coords[slot][0], coords[slot][1]
        contact = contact_or_none(x, y, kind)
        if contact is not None:
            out.append(contact)
    return tuple(out)


class CortexModel:
    """Runs cortex-sama frame by frame, keeping the streaming state."""

    name = "cortex"

    def __init__(
        self,
        model_dir: str | Path,
        ckpt: str | Path,
        *,
        device: str = "auto",
        input_size: tuple[int, int] = (320, 180),
        min_score: float = 0.0,
        warmup_frames: int = 16,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.ckpt = Path(ckpt)
        self.device_name = device
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.min_score = float(min_score)
        self.warmup_frames = int(warmup_frames)

        self._model = None
        self._torch = None
        self._device = None
        self._h = None
        self._cache = None
        self._stream_with_cache = False
        self.rejected = 0

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "CortexModel":
        """Import torch + the model repo and restore ``ckpt``."""
        if self._model is not None:
            return self

        torch = _import_torch()
        model_cls = self._import_model_class()
        model = model_cls()

        state = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

        self._device = self._pick_device(torch)
        model.to(self._device)
        model.eval()

        self._torch = torch
        self._model = model
        self._stream_with_cache = self._patch_tcn(model)
        return self

    def reset(self) -> None:
        """Forget the streaming history (new song / pause)."""
        self._h = None
        self._cache = None
        if self._model is not None:
            self._model.tcn._beatflow_cache = None

    def prime(self, frames: Sequence[Frame]) -> None:
        """Seed the GRU + TCN state from up to ``warmup_frames`` frames.

        Training always runs 16 warmup frames before a chunk, so a cold start
        is a distribution shift. A live run can skip this (the first frames
        after the game starts are ignored anyway); offline replay should use it.
        """
        self.load()
        frames = list(frames)[: self.warmup_frames]
        if not frames:
            return
        x = self._stack(frames)
        with self._torch.inference_mode():
            logits, _coords, h = self._run(x)
        self._h = h
        self._cache = self._model.tcn._beatflow_cache

    # -- model protocol ----------------------------------------------------
    def predict(self, frame: Frame) -> tuple[RawContact, ...]:
        self.load()
        x = self._to_tensor(frame.image)
        with self._torch.inference_mode():
            logits, coords, h = self._run(x)
        self._h = h
        self._cache = self._model.tcn._beatflow_cache

        last_logits = logits[0, -1]
        last_coords = coords[0, -1]
        probs = last_logits.softmax(-1)
        types = last_logits.argmax(-1).tolist()
        xy = last_coords.tolist()
        probability = probs.tolist()

        active = sum(1 for type_id in types if int(type_id) in _KIND)
        contacts = contacts_from_types(types, xy, probability, self.min_score)
        self.rejected += active - len(contacts)
        return contacts

    def describe(self) -> str:
        return (
            f"cortex: {self.input_size[0]}x{self.input_size[1]} rgb, "
            f"ckpt={self.ckpt}, device={self.device_name}, "
            f"min_score={self.min_score:g}"
        )

    # -- internals ---------------------------------------------------------
    def _run(self, x):
        """Forward one ``(B, T, 3, H, W)`` chunk, updating the TCN cache."""
        if self._stream_with_cache:
            logits, coords, h = self._model._forward(x, self._h, self._cache)
        else:
            # Older checkouts have no causal cache; fall back to the public API.
            if self._h is None:
                logits, coords, h = self._model(x)
            else:
                logits, coords, h = self._model(x, h=self._h)
        return logits, coords, h

    def _pick_device(self, torch):
        if self.device_name == "cpu":
            return torch.device("cpu")
        if self.device_name == "cuda":
            if not torch.cuda.is_available():
                print("warning: CUDA unavailable, using CPU", file=sys.stderr)
                return torch.device("cpu")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _import_model_class(self):
        model_py = self.model_dir / "model.py"
        if not model_py.is_file():
            raise FileNotFoundError(
                f"{model_py} not found; point model_dir at a cortex-sama checkout"
            )
        path = str(self.model_dir.resolve())
        sys.path.insert(0, path)
        try:
            module = importlib.import_module("model")
        finally:
            try:
                sys.path.remove(path)
            except ValueError:  # pragma: no cover - already removed
                pass

        module_path = Path(getattr(module, "__file__", "")).resolve()
        if not module_path.is_relative_to(self.model_dir.resolve()):
            raise RuntimeError(
                f"a different 'model' module is already imported ({module_path}); "
                "run the main program in a fresh process or rename the collision"
            )
        config = sys.modules.get("config")
        if config is not None and config.__file__ is not None:
            if not Path(config.__file__).resolve().is_relative_to(self.model_dir.resolve()):
                raise RuntimeError(
                    "a different top-level 'config' module is already imported; "
                    "cortex-sama's model.py does `import config as C` and would "
                    "read the wrong constants"
                )
        return module.Model

    @staticmethod
    def _patch_tcn(model) -> bool:
        """Keep ``CausalTCN``'s newest cache on the module.

        ``_forward`` takes ``tcn_cache`` but throws the updated one away. This
        wraps the module's ``forward`` so the adapter can read it after each
        call. Returns ``False`` for checkouts without a causal TCN.
        """
        tcn = getattr(model, "tcn", None)
        if tcn is None:
            return False
        if getattr(tcn, "_beatflow_patched", False):
            return True
        original = tcn.forward

        def forward_with_cache(x, cache=None):
            out, new_cache = original(x, cache)
            tcn._beatflow_cache = new_cache
            return out, new_cache

        tcn.forward = forward_with_cache  # type: ignore[method-assign]
        tcn._beatflow_cache = None
        tcn._beatflow_patched = True
        return True

    # -- image -> tensor ---------------------------------------------------
    def _to_tensor(self, image):
        torch = self._torch
        width, height = self.input_size

        if isinstance(image, torch.Tensor):
            t = image
        elif isinstance(image, (bytes, bytearray)):
            expected = width * height * 3
            if len(image) != expected:
                raise ValueError(
                    f"RGB bytes must be exactly {width}x{height}x3 = {expected} "
                    f"bytes, got {len(image)}"
                )
            t = torch.frombuffer(bytearray(image), dtype=torch.uint8)
            t = t.view(height, width, 3)
        elif hasattr(image, "__array__"):
            import numpy as np  # noqa: PLC0415 - torch already depends on it

            t = torch.from_numpy(np.asarray(image))
        else:
            raise TypeError(
                "Frame.image must be RGB bytes, an (H, W, 3) array or a tensor; "
                f"got {type(image).__name__}"
            )

        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        if t.dim() != 3:
            raise ValueError(f"expected a 3-channel image, got shape {tuple(t.shape)}")

        t = t.float()
        if t.max() > 1.0:
            t = t.div(255.0)
        if tuple(t.shape[-2:]) != (height, width):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(height, width),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return t.unsqueeze(0).unsqueeze(0).to(self._device)

    def _stack(self, frames: Sequence[Frame]):
        """Stack frames into ``(1, T, 3, H, W)`` for a warmup pass."""
        tensors = [self._to_tensor(frame.image).squeeze(0) for frame in frames]
        return self._torch.cat(tensors, dim=0).unsqueeze(0)
