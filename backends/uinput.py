"""Linux ``/dev/uinput`` backend - the real deal, zero third-party deps.

Implemented with :mod:`ctypes`/:mod:`fcntl` so the module runs on a bare
CPython install (this machine has no ``python-evdev`` and no ``pip``).

The device is exposed as a **direct-input** multitouch panel
(``INPUT_PROP_DIRECT``) speaking **MT protocol B**::

    ABS_MT_SLOT           which physical slot the following ABS_MT_* apply to
    ABS_MT_TRACKING_ID    new id == contact down, -1 == contact up
    ABS_MT_POSITION_X/Y   absolute position in device units

``BTN_TOUCH`` is toggled around the whole frame so legacy paths still see
"a finger is on the screen".

Notes for Wayland
-----------------
``uinput`` lives in the kernel, so it is compositor-agnostic: the event is
injected below libinput and the compositor routes the touch to whatever
surface is under ``(x, y)``. Two consequences:

* the game window **must be focused** - a struck-through window will not
  receive the touch;
* absolute device coordinates span the whole **output**, so letterboxed /
  windowed game areas need the mapping implemented in
  :mod:`beatflow_input.geometry`.
"""

from __future__ import annotations

import ctypes
import glob
import os
import struct
import time

try:  # Linux-only. Importing the package must still work on macOS/Windows so
    import fcntl  # the pure-logic half (labels/actions/geometry/visualizer) stays testable.
except ImportError:  # pragma: no cover - non-Linux development machines
    fcntl = None  # type: ignore[assignment]

from .base import BackendUnavailable, TouchBackend

__all__ = ["UinputBackend"]

# --- kernel constants -----------------------------------------------------
EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03

SYN_REPORT = 0

BTN_TOUCH = 0x14A  # 330

ABS_MT_SLOT = 0x2F
ABS_MT_TOUCH_MAJOR = 0x30
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
ABS_MT_TRACKING_ID = 0x39

INPUT_PROP_DIRECT = 0x01

BUS_VIRTUAL = 0x06


def _ioc(direction: int, typ: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(typ) << 8) | nr


def _io(typ: str, nr: int) -> int:
    return (ord(typ) << 8) | nr


def _iow(typ: str, nr: int, size: int) -> int:
    return _ioc(1, typ, nr, size)


UI_DEV_SETUP = _iow("U", 3, 92)  # struct uinput_setup
UI_ABS_SETUP = _iow("U", 4, 28)  # struct uinput_abs_setup
UI_DEV_CREATE = _io("U", 1)
UI_DEV_DESTROY = _io("U", 2)
UI_SET_EVBIT = _iow("U", 100, 4)
UI_SET_KEYBIT = _iow("U", 101, 4)
UI_SET_ABSBIT = _iow("U", 103, 4)
UI_SET_PROPBIT = _iow("U", 110, 4)

#: ``struct input_event`` on 64-bit Linux: timeval + u16 type + u16 code + s32.
_EVENT = struct.Struct("<qqHHi")


class UinputBackend(TouchBackend):
    """Multitouch device backed by ``/dev/uinput``.

    Parameters
    ----------
    capacity:
        Number of physical MT slots advertised (kernel max is 64; real panels
        usually expose 5-10). Keep this small: the slot allocator in front of
        this backend is what makes ``max_slots == 94`` label files work.
    dev_max_x, dev_max_y:
        Absolute coordinate range reported to the kernel. Anything is legal;
        32767 is the value every real digitizer uses.
    """

    name = "uinput"

    def __init__(
        self,
        capacity: int = 10,
        device_name: str = "BeatFlow Virtual Touch",
        dev_max_x: int = 32767,
        dev_max_y: int = 32767,
        vendor: int = 0x1209,
        product: int = 0xBEA7,
        version: int = 1,
        device_path: str = "/dev/uinput",
        create_timeout: float = 1.0,
        synthesize_btn_touch: bool = True,
    ) -> None:
        if not 1 <= capacity <= 64:
            raise ValueError("capacity must be in 1..64")
        self._capacity = capacity
        self.device_name = device_name
        self.dev_max_x = dev_max_x
        self.dev_max_y = dev_max_y
        self._vendor = vendor
        self._product = product
        self._version = version
        self.device_path = device_path
        self.create_timeout = create_timeout
        self.synthesize_btn_touch = synthesize_btn_touch

        self._fd: int | None = None
        self._node: str | None = None
        self._buf = bytearray()
        self._pos: dict[int, tuple[int, int]] = {}
        self._active: set[int] = set()
        self._btn_touch = False
        self._before_nodes: set[str] = set()
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)

    # -- properties --------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def node(self) -> str | None:
        """``/dev/input/eventN`` of the created device, if known."""
        return self._node

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if self._fd is not None:
            return
        if fcntl is None:  # pragma: no cover - non-Linux host
            raise BackendUnavailable(
                "the uinput backend needs Linux; use --backend dryrun on this host"
            )
        if not os.path.exists(self.device_path):
            raise BackendUnavailable(f"{self.device_path} does not exist")
        try:
            fd = os.open(self.device_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:  # pragma: no cover - depends on host setup
            raise BackendUnavailable(
                f"cannot open {self.device_path}: {exc}. "
                "Fix with a udev rule granting uaccess on /dev/uinput (see README)."
            ) from exc

        try:
            self._configure(fd)
        except OSError as exc:  # pragma: no cover
            os.close(fd)
            raise BackendUnavailable(f"uinput setup failed: {exc}") from exc

        self._fd = fd
        self._node = self._wait_for_node()

    def _configure(self, fd: int) -> None:
        self._before_nodes = set(glob.glob("/dev/input/event*"))

        for ev in (EV_SYN, EV_KEY, EV_ABS):
            fcntl.ioctl(fd, UI_SET_EVBIT, ev)
        fcntl.ioctl(fd, UI_SET_KEYBIT, BTN_TOUCH)
        for axis in (ABS_MT_SLOT, ABS_MT_TRACKING_ID, ABS_MT_POSITION_X,
                     ABS_MT_POSITION_Y, ABS_MT_TOUCH_MAJOR):
            fcntl.ioctl(fd, UI_SET_ABSBIT, axis)
        fcntl.ioctl(fd, UI_SET_PROPBIT, INPUT_PROP_DIRECT)

        absinfo = (
            (ABS_MT_SLOT, 0, self._capacity - 1),
            (ABS_MT_TRACKING_ID, 0, 65535),
            (ABS_MT_POSITION_X, 0, self.dev_max_x),
            (ABS_MT_POSITION_Y, 0, self.dev_max_y),
            (ABS_MT_TOUCH_MAJOR, 0, 255),
        )
        for axis, lo, hi in absinfo:
            # struct uinput_abs_setup { u16 code; input_absinfo absinfo; }
            fcntl.ioctl(fd, UI_ABS_SETUP, struct.pack("<H2xiiiiii", axis, 0, lo, hi, 0, 0, 0))

        # struct uinput_setup { input_id id; char name[80]; u32 ff_effects_max; }
        setup = struct.pack(
            "<HHHH80sI",
            BUS_VIRTUAL, self._vendor, self._product, self._version,
            self.device_name.encode("utf-8")[:79].ljust(80, b"\0"),
            0,
        )
        fcntl.ioctl(fd, UI_DEV_SETUP, setup)
        fcntl.ioctl(fd, UI_DEV_CREATE)

    def _wait_for_node(self) -> str | None:
        deadline = time.monotonic() + self.create_timeout
        while time.monotonic() < deadline:
            for node in set(glob.glob("/dev/input/event*")) - self._before_nodes:
                try:
                    with open(f"/sys/class/input/{os.path.basename(node)}/device/name") as fh:
                        if fh.read().strip() == self.device_name:
                            return node
                except OSError:
                    continue
            time.sleep(0.01)
        return None

    def close(self) -> None:
        if self._fd is None:
            return
        try:
            for slot in sorted(self._active):
                self.release(slot)
            if self._buf or self._active:
                self.commit()
            elif self._btn_touch:
                self.commit()
        except OSError:
            pass
        finally:
            try:
                fcntl.ioctl(self._fd, UI_DEV_DESTROY)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None
            self._node = None
            self._active.clear()
            self._pos.clear()
            self._btn_touch = False
            self._buf.clear()

    # -- event queue -------------------------------------------------------
    def _emit(self, ev_type: int, code: int, value: int) -> None:
        self._buf += _EVENT.pack(0, 0, ev_type, code, value)

    def press(self, slot: int, tracking_id: int, x: int, y: int) -> None:
        if not 0 <= slot < self._capacity:
            raise ValueError(f"slot {slot} outside 0..{self._capacity - 1}")
        self._emit(EV_ABS, ABS_MT_SLOT, slot)
        self._emit(EV_ABS, ABS_MT_TRACKING_ID, tracking_id & 0xFFFF)
        self._emit(EV_ABS, ABS_MT_POSITION_X, x)
        self._emit(EV_ABS, ABS_MT_POSITION_Y, y)
        self._pos[slot] = (x, y)
        self._active.add(slot)

    def move(self, slot: int, x: int, y: int) -> None:
        if self._pos.get(slot) == (x, y):
            return
        self._emit(EV_ABS, ABS_MT_SLOT, slot)
        self._emit(EV_ABS, ABS_MT_POSITION_X, x)
        self._emit(EV_ABS, ABS_MT_POSITION_Y, y)
        self._pos[slot] = (x, y)

    def release(self, slot: int) -> None:
        if slot not in self._active:
            return
        self._emit(EV_ABS, ABS_MT_SLOT, slot)
        self._emit(EV_ABS, ABS_MT_TRACKING_ID, -1)
        self._active.discard(slot)
        self._pos.pop(slot, None)

    def commit(self) -> None:
        if self._fd is None:
            raise BackendUnavailable("backend is not open")

        if self.synthesize_btn_touch:
            want = bool(self._active)
            if want != self._btn_touch:
                frame = _EVENT.pack(0, 0, EV_KEY, BTN_TOUCH, 1 if want else 0)
                if want:
                    self._buf[:0] = frame
                else:
                    self._buf += frame
                self._btn_touch = want

        self._buf += _EVENT.pack(0, 0, EV_SYN, SYN_REPORT, 0)
        # Copy out: a memoryview over the bytearray would block resizing it.
        data = bytes(self._buf)
        self._buf.clear()
        written = 0
        try:
            while written < len(data):
                written += os.write(self._fd, data[written:])
        except OSError as exc:  # pragma: no cover
            raise BackendUnavailable(f"uinput write failed: {exc}") from exc

    def status(self) -> dict:
        """Human-readable snapshot used by the CLI."""
        return {
            "backend": self.name,
            "node": self._node,
            "device_name": self.device_name,
            "capacity": self._capacity,
            "abs_range": [self.dev_max_x, self.dev_max_y],
            "active_slots": sorted(self._active),
        }
