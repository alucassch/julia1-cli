/* C API of the Julia-1 engine (engine.h) for other languages: JSON strings in and out, no C++ types across the boundary.
 * Built as the shared library julia1_gguf_native (CMake JULIA1_BUILD_SHARED) and installed as julia1_gguf.h;
 * julia1_gguf/native.py loads it with ctypes.
 *
 * Strings returned as a result or through *error are allocated by the library: release them with julia1_string_free.
 * Calls on one handle are serialised by a mutex (any thread may call); different handles are independent.
 *
 * ggml's log is quiet: only its warnings and errors reach stderr. Environment variable JULIA1_GGUF_VERBOSE=1 (read when
 * the first engine is created) keeps ggml's full log (the Metal device and library lines that julia1-cli prints). */
#ifndef JULIA1_C_API_H
#define JULIA1_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* options_json: {"model": path (required), "device": "auto"|"cpu"|"metal", "threads": 4, "precise": true,
 * "max_length": 8192 (null: the model's context length), "head_length": 256, "strict": false, "batch": 1}, the fields
 * and defaults of EngineOptions (engine.h). Returns a handle, or NULL with *error set (error may be NULL). */
void * julia1_engine_new(const char * options_json, char ** error);

/* Runs one method on request_json and returns its result as JSON, or NULL with *error set:
 *   info           (request ignored)                    -> Engine::info()
 *   logits         [request, ...]                       -> [[logit, ...], ...]          upstream engine.logits(rows)
 *   predict_rows   [request, ...]                       -> [{"index", "probabilities"}]  upstream engine.predict(rows)
 *   predict_typed  {"state": ..., "questions": {...}}   -> {"answers": {...}}           upstream engine.predict(state=, questions=)
 *   replay         [{"ids", "markers", "qtype"}, ...]   -> [[logit, ...], ...]          already-encoded sequences
 *   encode         [request, ...]                       -> [{"ids", "markers", "qtype"}] SPEC §5 encoding
 *   tokenize       [text, ...]                          -> [[token id, ...], ...]       SPEC §4, without CLS/SEP
 * A request is {"state", "question", "options", "type"} (upstream julia/data.py); errors carry upstream's messages.
 * Results are serialised like julia1-cli's output lines. */
char * julia1_engine_call(void * engine, const char * method, const char * request_json, char ** error);

void julia1_string_free(char * s);
void julia1_engine_free(void * engine);

/* "julia1-engine <version> (ggml <version>[, exact kernels])"; a static string, not to be freed. */
const char * julia1_version(void);

#ifdef __cplusplus
}
#endif

#endif
