cmake_minimum_required(VERSION 3.18) # FetchContent_MakeAvailable with SOURCE_SUBDIR, find_program REQUIRED
project(julia1-gguf VERSION 0.2.0 LANGUAGES C CXX) # the release version: julia1-cli --version, julia1_version(), info()
set(JULIA1_VERSION ${PROJECT_VERSION})              # (pyproject.toml repeats it; tools/package_release.sh checks both)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# __FILE__ (ggml's assert messages) and debug info name sources relative to the source and build trees, so the
# binaries carry no absolute paths of the build machine
if (NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=>"
                        "$<$<COMPILE_LANGUAGE:C,CXX,OBJC,OBJCXX>:-ffile-prefix-map=${CMAKE_BINARY_DIR}/=>")
endif()

# Targets: julia1-cli (JULIA1_BUILD_CLI: the CLI and HTTP server) and the shared library julia1_gguf_native
# (JULIA1_BUILD_SHARED: the C API of src/c_api.h, which julia1_gguf/native.py loads; the Python build, pyproject.toml,
# builds only this one). Both link the same runtime objects (julia1-core) and the same static ggml.
option(JULIA1_BUILD_CLI    "build julia1-cli (CLI and HTTP server)" ON)
option(JULIA1_BUILD_SHARED "build the shared library julia1_gguf_native (C API)" ON)
if (JULIA1_BUILD_SHARED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # the static ggml and julia1-core go into a shared library
endif()

# ggml from the llama.cpp v0.5.0 release (below): static libraries, Metal on Apple, no tests/examples.
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# JULIA1_BLAS (default on Apple): ggml's BLAS backend (Accelerate) for the large matmuls of the CPU device, which then
# runs the graph over [BLAS, CPU] through ggml_backend_sched; Metal never uses it. With the patched ggml, F32 weights
# and --threads + 2 <= the performance cores, the CPU backend calls Accelerate itself instead. Elsewhere it is off;
# -DJULIA1_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS selects OpenBLAS.
option(JULIA1_BLAS "CPU device: BLAS backend (Accelerate on Apple) for the large matmuls" ${APPLE})
set(GGML_BLAS           ${JULIA1_BLAS} CACHE BOOL "" FORCE)
# CPU backend: llamafile's register-tiled sgemm (tinyBLAS) for the matmuls instead of per-row dot products.
set(GGML_LLAMAFILE      ON  CACHE BOOL "" FORCE)
if (APPLE)
    set(GGML_METAL               ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
# ggml and the vendored headers (nlohmann json, cpp-httplib) come from the llama.cpp v0.5.0 release tarball (ggml
# 0.25.1), downloaded at configure time and checked against its SHA-256. An unpacked copy of that release can be used
# instead, offline: -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<absolute path> (the tree is only read, never modified).
# JULIA1_GGML_PATCH (default ON): build a copy of its ggml/, made in the build directory, with patches/ggml-julia1.patch
# applied: exact float-staged Metal matmul and flash-attention kernels, which julia1-cli detects through
# GGML_JULIA1_EXACT_MM in ggml.h, fused Metal GEMM epilogues, and CPU speedups. OFF builds the stock ggml (same outputs
# in --fast and on the CPU up to float noise; --precise falls back to a ~4x slower exact path).
# JULIA1_GGML_DIR / JULIA1_VENDOR_DIR (relative to this directory or absolute): use these trees as they are instead; the
# download is skipped when both are set.
option(JULIA1_GGML_PATCH "apply patches/ggml-julia1.patch to the fetched ggml" ON)
set(JULIA1_GGML_DIR   "" CACHE STRING "ggml source tree used as is instead of the fetched one")
set(JULIA1_VENDOR_DIR "" CACHE STRING "directory with nlohmann/ and cpp-httplib/ (default: the fetched llama.cpp's vendor/)")
if (NOT JULIA1_GGML_DIR OR NOT JULIA1_VENDOR_DIR)
    if (POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW) # extracted files get the extraction time
    endif()
    include(FetchContent)
    FetchContent_Declare(llama_cpp
        URL           https://github.com/ggml-org/llama.cpp/archive/refs/tags/v0.5.0.tar.gz
        URL_HASH      SHA256=fef9ed754f4e031fb5c663c29260feda4ebc241abb68d64a81c0f1df5f1748e2
        SOURCE_SUBDIR ggml/include) # has no CMakeLists.txt: populate only, llama.cpp itself is not added to the build
    FetchContent_MakeAvailable(llama_cpp)
endif()

if (JULIA1_GGML_DIR)
    get_filename_component(julia1_ggml_dir ${JULIA1_GGML_DIR} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(julia1_ggml_kind "JULIA1_GGML_DIR, as is")
elseif (JULIA1_GGML_PATCH)
    # (re)made when the patch or the source tree changes; the stamp records both
    set(julia1_ggml_dir ${CMAKE_BINARY_DIR}/ggml-julia1)
    set(julia1_patch ${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-julia1.patch)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${julia1_patch})
    file(SHA256 ${julia1_patch} julia1_patch_sha256)
    set(julia1_stamp "${llama_cpp_SOURCE_DIR}/ggml ${julia1_patch_sha256}")
    set(julia1_old_stamp "")
    if (EXISTS ${julia1_ggml_dir}/.julia1-patch-stamp)
        file(READ ${julia1_ggml_dir}/.julia1-patch-stamp julia1_old_stamp)
    endif()
    if (NOT julia1_old_stamp STREQUAL julia1_stamp)
        find_program(JULIA1_PATCH_EXECUTABLE patch REQUIRED)
        file(REMOVE_RECURSE ${julia1_ggml_dir})
        file(COPY ${llama_cpp_SOURCE_DIR}/ggml/ DESTINATION ${julia1_ggml_dir})
        execute_process(COMMAND ${JULIA1_PATCH_EXECUTABLE} -p2 -N -F0 -s -i ${julia1_patch} # -F0: no fuzzy hunks
                        WORKING_DIRECTORY ${julia1_ggml_dir} RESULT_VARIABLE julia1_rc OUTPUT_VARIABLE julia1_out ERROR_VARIABLE julia1_out)
        if (NOT julia1_rc EQUAL 0)
            file(REMOVE_RECURSE ${julia1_ggml_dir})
            message(FATAL_ERROR "applying ${julia1_patch} to ${llama_cpp_SOURCE_DIR}/ggml failed:\n${julia1_out}")
        endif()
        file(WRITE ${julia1_ggml_dir}/.julia1-patch-stamp ${julia1_stamp})
    endif()
    set(julia1_ggml_kind "llama.cpp v0.5.0 + patches/ggml-julia1.patch")
else()
    set(julia1_ggml_dir ${llama_cpp_SOURCE_DIR}/ggml)
    set(julia1_ggml_kind "llama.cpp v0.5.0, stock")
endif()
if (JULIA1_VENDOR_DIR)
    get_filename_component(julia1_vendor_dir ${JULIA1_VENDOR_DIR} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
    set(julia1_vendor_dir ${llama_cpp_SOURCE_DIR}/vendor)
endif()
if (NOT EXISTS ${julia1_ggml_dir}/CMakeLists.txt OR NOT EXISTS ${julia1_vendor_dir}/cpp-httplib/httplib.cpp)
    message(FATAL_ERROR "ggml (${julia1_ggml_dir}) or vendor (${julia1_vendor_dir}) tree not found")
endif()
message(STATUS "julia1-cli: ggml from ${julia1_ggml_dir} (${julia1_ggml_kind}); vendor from ${julia1_vendor_dir}")
# EXCLUDE_FROM_ALL: ggml builds only what julia1-core links, and its own install rules (static libraries, headers,
# CMake package) stay out of cmake --install
add_subdirectory(${julia1_ggml_dir} ggml EXCLUDE_FROM_ALL)

add_library(julia1-core OBJECT src/encoding.cpp src/engine.cpp src/gguf_model.cpp src/graph.cpp src/tokenizer.cpp)
target_include_directories(julia1-core PUBLIC ${julia1_vendor_dir})
# `ggml` links ggml-base and every enabled backend (PUBLIC) and defines GGML_USE_<BACKEND>,
# so the static backends register themselves; no ggml_backend_load_all() is needed.
target_link_libraries(julia1-core PUBLIC ggml)
target_compile_definitions(julia1-core PRIVATE JULIA1_ENGINE_VERSION="${JULIA1_VERSION}")

if (JULIA1_BUILD_CLI)
    # cpp-httplib (vendored inside llama.cpp's vendor/, MIT) for julia1-cli serve, built here: its own CMakeLists.txt
    # needs llama.cpp's build functions. No TLS (put the server behind a reverse proxy). The form-urlencoded limit (8 KB
    # by default) is lifted so that a JSON body sent without a Content-Type (curl -d) only meets --max-body-mb.
    find_package(Threads REQUIRED)
    add_library(cpp-httplib STATIC ${julia1_vendor_dir}/cpp-httplib/httplib.cpp)
    target_include_directories(cpp-httplib PUBLIC ${julia1_vendor_dir})
    target_compile_definitions(cpp-httplib PRIVATE CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=SIZE_MAX CPPHTTPLIB_LISTEN_BACKLOG=512)
    if (NOT MSVC)
        target_compile_options(cpp-httplib PRIVATE -w) # third-party warnings
    endif()
    target_link_libraries(cpp-httplib PUBLIC Threads::Threads)

    add_executable(julia1-cli src/main.cpp src/server.cpp)
    target_link_libraries(julia1-cli PRIVATE julia1-core cpp-httplib)
endif()

if (JULIA1_BUILD_SHARED)
    add_library(julia1_gguf_native SHARED src/c_api.cpp)
    target_link_libraries(julia1_gguf_native PRIVATE julia1-core)
    # pip install . (scikit-build-core) installs this component only, into the julia1_gguf package
    install(TARGETS julia1_gguf_native DESTINATION julia1_gguf COMPONENT python EXCLUDE_FROM_ALL)
endif()

# cmake --install <build> [--prefix P]: the release layout (tools/package_release.sh):
#   bin/julia1-cli, lib/libjulia1_gguf_native.*, include/julia1_gguf.h (src/c_api.h),
#   share/julia1-cli/{README-quickstart.md, LICENSE, NOTICE, THIRD_PARTY_LICENSES}
include(GNUInstallDirs)
if (JULIA1_BUILD_CLI)
    install(TARGETS julia1-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if (JULIA1_BUILD_SHARED)
    install(TARGETS julia1_gguf_native LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES src/c_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} RENAME julia1_gguf.h)
endif()
install(FILES packaging/README-quickstart.md LICENSE NOTICE THIRD_PARTY_LICENSES DESTINATION ${CMAKE_INSTALL_DATADIR}/julia1-cli)
