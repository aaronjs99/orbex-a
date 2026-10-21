"""Spawned frozen-source bootstrap and isolated solver descendant lifetime."""
from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import os
import signal
import sys
from orbexa.simulation.recording.artifacts import _write_json


class _WorkerTree:
    """Windows job lifetime includes APMonitor descendants, not other jobs."""

    def __init__(self, process):
        self.process, self.handle, self.kernel = process, None, None
        if os.name != "nt":
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_mem", ctypes.c_size_t),
                ("job_mem", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes, kernel.CreateJobObjectW.restype = [ctypes.c_void_p, w.LPCWSTR], w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
        kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        information = Extended()
        information.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        child = kernel.OpenProcess(0x0101, False, process.pid)
        try:
            if not handle or not child or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(information), ctypes.sizeof(information)) or not kernel.AssignProcessToJobObject(handle, child):
                raise OSError(ctypes.get_last_error(), "Cannot isolate the live solver process tree")
        except BaseException:
            if handle:
                kernel.CloseHandle(handle)
            raise
        finally:
            if child:
                kernel.CloseHandle(child)
        self.handle, self.kernel = handle, kernel

    def close(self, *, cancel=False):
        if self.handle is not None:
            if cancel:
                self.kernel.TerminateJobObject(self.handle, 1)
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif cancel and self.process.is_alive():
            if os.name != "nt":
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.process.terminate()
            else:
                self.process.terminate()
        self.process.join(3)
        if cancel and self.process.is_alive():
            self.process.kill()
            self.process.join(3)


def _run_job(settings, parameters, directory, messages, run_event, launch_event):
    if os.name != "nt":
        os.setsid()
    launch_event.wait()
    with (directory / "worker.log").open("a", encoding="utf-8", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            frozen = directory / 'runtime' / 'orbexa'
            for name in tuple(sys.modules):
                if name == 'orbexa' or name.startswith('orbexa.'):
                    del sys.modules[name]
            spec = importlib.util.spec_from_file_location('orbexa', frozen / '__init__.py',
                submodule_search_locations=[str(frozen)])
            package = importlib.util.module_from_spec(spec)
            sys.modules['orbexa'] = package
            spec.loader.exec_module(package)
            from orbexa.simulation.service.jobs.dispatch import _execute_job as frozen_execute
            frozen_execute(settings, parameters, directory, messages, run_event)
        except Exception as error:
            import traceback
            traceback.print_exc()
            payload = dict(kind="error", success=False, message=f"{type(error).__name__}: {error}")
            _write_json(directory / "worker-result.json", payload)
            messages.put(payload)
