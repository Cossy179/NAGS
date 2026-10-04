cmake_minimum_required(VERSION 3.15)

project(NAGS_UCI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Makefiles, Ninja) default to an optimised build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NAGS_BUILD_TESTS "Build the C++ tests (run with ctest)" ON)

find_package(Threads REQUIRED)

function(nags_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

# Syzygy tablebase probing (third_party/fathom, MIT). Compiled as C++ so it
# needs no C11 atomics support (MSVC).
add_library(fathom STATIC third_party/fathom/tbprobe.c)
set_source_files_properties(third_party/fathom/tbprobe.c PROPERTIES LANGUAGE CXX)
target_include_directories(fathom PUBLIC third_party/fathom)
target_link_libraries(fathom PUBLIC Threads::Threads)
if(MSVC)
  target_compile_definitions(fathom PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Default NNUE network, compiled into the engines (see tools/nnue/train.py).
# Without the file the engines use the hand-written evaluation.
set(NAGS_NNUE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/nets/nags.nnue" CACHE FILEPATH "NNUE network to embed")
set(NAGS_EMBEDDED_NET ${CMAKE_CURRENT_BINARY_DIR}/nnue_embedded.cpp)
set(embed_depends ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_net.cmake)
if(EXISTS "${NAGS_NNUE_FILE}")
  list(APPEND embed_depends "${NAGS_NNUE_FILE}")
endif()
add_custom_command(OUTPUT ${NAGS_EMBEDDED_NET}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${NAGS_NNUE_FILE} -DOUTPUT=${NAGS_EMBEDDED_NET}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_net.cmake
  DEPENDS ${embed_depends}
  COMMENT "Embedding NNUE network")

# Code shared by every engine: boards, transposition table, UCI front end,
# tablebases, NNUE, networking for the Python services.
add_library(nags_core STATIC
  src/Board.cpp
  src/FastBoard.cpp
  src/TT.cpp
  src/Uci.cpp
  src/Syzygy.cpp
  src/Nnue.cpp
  ${NAGS_EMBEDDED_NET}
  src/Net.cpp
  src/MetaClient.cpp
)
target_include_directories(nags_core PUBLIC src)
target_link_libraries(nags_core PUBLIC Threads::Threads PRIVATE fathom)
if(WIN32)
  target_link_libraries(nags_core PUBLIC ws2_32)
endif()
nags_warnings(nags_core)

# Alpha-beta on the ray-based Board, no hash table.
add_executable(nags_basic src/basic_main.cpp)
# Alpha-beta on the magic-bitboard FastBoard, no hash table.
add_executable(nags_fast src/fast_main.cpp)
# FastBoard + shared transposition table + Lazy SMP threads.
add_executable(nags_enhanced src/enhanced_main.cpp)
# NAGS hybrid: nags_enhanced's alpha-beta plus a GNN-guided MCTS arm (over RPC)
# whose proposals alpha-beta verifies; meta-learner adjusts it per move.
add_executable(nags src/main.cpp src/NAGS.cpp)
# Smoke test client for rpc_server.py.
add_executable(rpc_client src/rpc_client.cpp)
# Self-play training data for the NNUE evaluation.
add_executable(nags_datagen src/datagen_main.cpp)

foreach(target nags_basic nags_fast nags_enhanced nags rpc_client nags_datagen)
  target_link_libraries(${target} PRIVATE nags_core)
  nags_warnings(${target})
endforeach()

if(NAGS_BUILD_TESTS)
  enable_testing()

  add_executable(board_tests tests/cpp/board_tests.cpp)
  add_executable(fastboard_tests tests/cpp/board_tests.cpp)
  target_compile_definitions(fastboard_tests PRIVATE TEST_FAST_BOARD)
  add_executable(search_tests tests/cpp/search_tests.cpp)
  add_executable(syzygy_tests tests/cpp/syzygy_tests.cpp)
  add_executable(nnue_tests tests/cpp/nnue_tests.cpp)
  set(NAGS_SYZYGY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/syzygy)
  target_compile_definitions(syzygy_tests PRIVATE NAGS_SYZYGY_DIR="${NAGS_SYZYGY_DIR}")
  foreach(target board_tests fastboard_tests search_tests syzygy_tests nnue_tests)
    target_link_libraries(${target} PRIVATE nags_core)
    nags_warnings(${target})
    add_test(NAME ${target} COMMAND ${target})
  endforeach()

  # End-to-end UCI tests: feed a command script to an engine and check its output.
  function(nags_uci_test name engine input expect)
    add_test(NAME ${name}
      COMMAND ${CMAKE_COMMAND}
        -DENGINE=$<TARGET_FILE:${engine}>
        -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/uci/${input}
        -DEXPECT=${expect}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/uci/run_uci_test.cmake)
  endfunction()

  foreach(engine nags_basic nags_fast nags_enhanced nags)
    nags_uci_test(${engine}_handshake ${engine} handshake.txt "uciok.*readyok")
    # Black to move after 1.e4: the reply must be a black move.
    nags_uci_test(${engine}_side_to_move ${engine} black_to_move.txt "bestmove [a-h][78][a-h][56]")
    nags_uci_test(${engine}_mate_in_one ${engine} mate_in_one.txt "score mate 1 .*bestmove f3f7")
    nags_uci_test(${engine}_bad_input ${engine} bad_input.txt "invalid FEN.*bestmove [a-h][1-8][a-h][1-8]")
    nags_uci_test(${engine}_movetime ${engine} movetime.txt "bestmove [a-h][1-8][a-h][1-8]")
  endforeach()
  nags_uci_test(nags_enhanced_threads nags_enhanced threads.txt "Threads set to 4.*bestmove [a-h][1-8][a-h][1-8]")
  configure_file(tests/uci/syzygy.txt.in ${CMAKE_CURRENT_BINARY_DIR}/syzygy_uci.txt @ONLY)
  foreach(engine nags_basic nags_fast nags_enhanced nags)
    add_test(NAME ${engine}_syzygy
      COMMAND ${CMAKE_COMMAND}
        -DENGINE=$<TARGET_FILE:${engine}>
        -DINPUT=${CMAKE_CURRENT_BINARY_DIR}/syzygy_uci.txt
        "-DEXPECT=tablebases loaded: up to 3 pieces.*bestmove [a-h][1-8][a-h][1-8]"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/uci/run_uci_test.cmake)
  endforeach()
  foreach(engine nags_basic nags_fast nags_enhanced)
    nags_uci_test(${engine}_multipv ${engine} multipv.txt "depth 5 [^ ]+ [^ ]+ multipv 3 .*bestmove")
    nags_uci_test(${engine}_ponder ${engine} ponder.txt "bestmove [a-h][78][a-h][56]")
  endforeach()

  # Without its Python services, nags must search exactly like nags_enhanced.
  add_test(NAME nags_equals_nags_enhanced
    COMMAND ${CMAKE_COMMAND} -DENGINE=$<TARGET_FILE:nags> -DENGINE2=$<TARGET_FILE:nags_enhanced> -DDEPTH=6
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/uci/run_bench_equal.cmake)

  # The bench node count is the search fingerprint: it must be reproducible.
  foreach(engine nags_basic nags_fast nags_enhanced nags)
    add_test(NAME ${engine}_bench_deterministic
      COMMAND ${CMAKE_COMMAND} -DENGINE=$<TARGET_FILE:${engine}> -DDEPTH=4
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/uci/run_bench_test.cmake)
  endforeach()
endif()
