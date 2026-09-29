"""Julia-1 decision model in GGUF: native C++/ggml runtime, numpy runtime, SPEC §4 tokenizer, SPEC §5 encoding.

`load_model(path)` returns an engine mirroring upstream `julia.load_model`.
Imports are lazy so `python -m julia1_gguf --threads N` can set BLAS thread
environment variables before numpy loads.
"""

__version__ = '0.1.0'

_warned = False


def load_model(path, max_length=None, head_length=256, strict_encoding=False, *, backend='auto', device='auto',
               precise=True, threads=4, batch=1, tokenizer='auto'):
    """Upstream `julia.load_model` defaults (max_length=None -> the model's context length, head_length=256, strict
    off); the evaluation protocol uses max_length=1024, head_length=512, strict_encoding=True.

    backend: 'native' (julia1-cli's C++/ggml runtime, native.NativeEngine; device auto|cpu|metal, precise: exact F32 on
    Metal or False for half-precision operands, threads: CPU threads, batch: requests per padded graph), 'numpy'
    (engine.Engine, CPU; tokenizer auto|hf|pure), or 'auto': native when its library loads, else numpy with a warning.
    """
    global _warned
    if backend not in ('auto', 'native', 'numpy'):
        raise ValueError('backend must be auto, native or numpy')
    if path is None:
        raise ValueError('path is required; pass a Julia-1 GGUF file')
    if backend != 'numpy':
        from . import native
        try:
            native.load_library()
        except (ImportError, OSError) as error:
            if backend == 'native':
                raise
            if not _warned:
                import warnings
                warnings.warn(f'julia1_gguf: native library unavailable ({error}); using the numpy backend, ~12x slower',
                              RuntimeWarning, stacklevel=2)
                _warned = True
        else:
            return native.NativeEngine(path, max_length, head_length, strict_encoding, device=device, precise=precise,
                                       threads=threads, batch=batch)
    from .engine import load_model as _load
    return _load(path, max_length, head_length, strict_encoding, tokenizer=tokenizer)


__all__ = ['load_model']
