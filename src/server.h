// julia1-cli serve: HTTP API over one Engine with dynamic batching (docs/server.md).
#pragma once

#include "engine.h"

#include <string>

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;           // 0: any free port (the "serving on" line names it)
    std::string api_key;       // empty: no authentication
    int max_batch = 16;        // sequences per batch; the Engine must be created with EngineOptions::batch = max_batch
    double batch_wait_ms = -1; // how long the oldest queued job waits for more; < 0: 0 on a GPU, 2 on the CPU
    int max_body_mb = 4;
    int http_threads = 8;      // HTTP worker threads (cpp-httplib grows the pool up to 4x under load)
    bool quiet = false;        // no per-request log line
};

// Serves until SIGINT/SIGTERM, then answers the requests in flight and returns the process exit code.
// Throws std::runtime_error when the address cannot be bound or accepting connections fails.
// helper (--devices metal,cpu): a second Engine on the same model, created with EngineOptions::batch = max_batch; it
// takes a batch only while `engine` runs one and at least max_batch sequences are queued.
int serve(Engine & engine, const ServerOptions & options, Engine * helper = nullptr);
