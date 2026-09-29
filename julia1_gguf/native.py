"""The C++/ggml runtime of julia1-cli (Metal or CPU) through its C API (src/c_api.h) and ctypes.

`NativeEngine` has the methods and return types of the numpy `engine.Engine`: logits(rows), predict(rows) (legacy list
API), predict(state=..., questions=...) (named questions), replay(ids, markers, qtype), encode(row), tokenize(text).
Requests are validated here first, by the numpy engine's code (upstream's checks of Python types: a tuple is not a
list, a question id must be a str), then sent as JSON (`json.dumps`), so any other key they carry must be
JSON-serialisable.

The library (libjulia1_gguf_native.dylib / .so) is JULIA1_GGUF_LIB when set, else the one `pip install .` puts inside
this package. No numpy import: `load_model(backend='auto')` tries this module first. ggml logs only its warnings and
errors (stderr); JULIA1_GGUF_VERBOSE=1, set before the first engine is created, shows its full log.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys

from .encoding import typed_rows, validate_row

LIBRARY = {'darwin': 'libjulia1_gguf_native.dylib', 'win32': 'julia1_gguf_native.dll'}.get(sys.platform, 'libjulia1_gguf_native.so')
_lib = None


def library_path():
    """JULIA1_GGUF_LIB, else the library in one of the package's directories (an editable install keeps the built
    library in site-packages, apart from the sources: both are on the package's __path__), else the installed
    julia1-gguf's (Python started in a source checkout imports the checkout's julia1_gguf, which has no library)."""
    if os.environ.get('JULIA1_GGUF_LIB'):
        return os.environ['JULIA1_GGUF_LIB']
    dirs = list(sys.modules[__package__].__path__)
    for d in dirs:
        path = os.path.join(d, LIBRARY)
        if os.path.isfile(path):
            return path
    from importlib import metadata
    for dist in metadata.distributions(name='julia1-gguf'):  # all of them: a checkout may hold a stale *.egg-info
        path = str(dist.locate_file(f'julia1_gguf/{LIBRARY}'))
        if os.path.isfile(path):
            return path
    raise ImportError(f'{LIBRARY} not found in {dirs} or an installed julia1-gguf (pip install . builds it; or set '
                      'JULIA1_GGUF_LIB)')


def load_library():
    """The ctypes library, loaded once (ImportError or OSError when it is missing or cannot load)."""
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(library_path())  # CDLL releases the GIL during calls; the library serialises calls per engine
        char_pp = ctypes.POINTER(ctypes.c_void_p)  # char ** out-parameter; results stay c_void_p so they can be freed
        lib.julia1_engine_new.argtypes = [ctypes.c_char_p, char_pp]
        lib.julia1_engine_new.restype = ctypes.c_void_p
        lib.julia1_engine_call.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, char_pp]
        lib.julia1_engine_call.restype = ctypes.c_void_p
        lib.julia1_string_free.argtypes = [ctypes.c_void_p]
        lib.julia1_string_free.restype = None
        lib.julia1_engine_free.argtypes = [ctypes.c_void_p]
        lib.julia1_engine_free.restype = None
        lib.julia1_version.argtypes = []
        lib.julia1_version.restype = ctypes.c_char_p
        _lib = lib
    return _lib


def version():
    return load_library().julia1_version().decode()


def _take(lib, pointer):
    """The library's string at pointer, released."""
    try:
        return ctypes.string_at(pointer)
    finally:
        lib.julia1_string_free(pointer)


class NativeEngine:
    backend = 'native'

    def __init__(self, path, max_length=None, head_length=256, strict_encoding=False, *, device='auto', precise=True,
                 threads=4, batch=1):
        self._handle = None
        self._lib = load_library()
        path = os.fspath(path)
        if not os.path.isfile(path):  # the numpy engine's error for a missing file
            raise FileNotFoundError(2, 'No such file or directory', path)
        options = dict(model=path, device=device, precise=precise, threads=threads, max_length=max_length,
                       head_length=head_length, strict=strict_encoding, batch=batch)
        error = ctypes.c_void_p()
        self._handle = self._lib.julia1_engine_new(json.dumps(options).encode(), ctypes.byref(error))
        if not self._handle:
            raise ValueError(_take(self._lib, error.value).decode())
        self.info = self._call('info')  # device, mode, threads, load_ms, version, ... (Engine::info)
        self.max_length, self.head_length, self.strict = self.info['max_length'], head_length, strict_encoding

    def _call(self, method, request=None):
        text = None if request is None else json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
        error = ctypes.c_void_p()
        result = self._lib.julia1_engine_call(self._handle, method.encode(), text, ctypes.byref(error))
        if not result:
            raise ValueError(_take(self._lib, error.value).decode())
        return json.loads(_take(self._lib, result))

    def close(self):
        """Frees the engine (also on garbage collection)."""
        if self._handle:
            self._lib.julia1_engine_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def encode(self, row):
        """SPEC §5 encoding of one request -> {ids, markers, qtype}."""
        return self._call('encode', [row])[0]

    def tokenize(self, text):
        """SPEC §4 token ids of a text, without CLS/SEP."""
        return self._call('tokenize', [text])[0]

    def replay(self, ids, markers, qtype):
        """Logits for an already-encoded sequence."""
        return self._call('replay', [dict(ids=list(map(int, ids)), markers=list(map(int, markers)), qtype=int(qtype))])[0]

    def logits(self, rows):
        if not rows:
            return []
        rows = list(rows)
        for i, row in enumerate(rows):
            validate_row(row, i + 1)
        return self._call('logits', rows)

    def predict(self, rows=None, questions=None, *, state=None):
        if questions is not None:
            if rows is not None and state is not None:
                raise ValueError('Pass state either positionally or by keyword, not both')
            state = state if rows is None else rows
            typed_rows(state, questions)
            return self._call('predict_typed', dict(state=state, questions=questions))
        if state is not None or rows is None:
            raise ValueError('Provide legacy rows or state with questions')
        rows = list(rows)
        for i, row in enumerate(rows):
            validate_row(row, i + 1)
        return self._call('predict_rows', rows)
