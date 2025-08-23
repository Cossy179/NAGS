cmake_minimum_required(VERSION 3.15)

project(NAGS_UCI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find threading library
find_package(Threads REQUIRED)

# Basic UCI engine with random move selection
add_executable(nags_basic
  src/basic_main.cpp
  src/Board.h
  src/Board.cpp
)

# Fast bitboard UCI engine
add_executable(nags_fast
  src/fast_main.cpp
  src/FastBoard.h
  src/FastBoard.cpp
)

# Enhanced UCI engine with TT and threading
add_executable(nags_enhanced
  src/enhanced_main.cpp
  src/FastBoard.h
  src/FastBoard.cpp
  src/FastTT.h
  src/FastTT.cpp
)

# Link threading library
target_link_libraries(nags_enhanced Threads::Threads)

# Advanced NAGS hybrid engine
add_executable(nags
  src/main.cpp
  src/Board.h
  src/Board.cpp
  src/Search.h
  src/Search.cpp
  src/TT.h
  src/TT.cpp
  src/NAGS.h
  src/NAGS.cpp
  src/MetaClient.h
  src/MetaClient.cpp
)

add_executable(rpc_client
  src/rpc_client.cpp
)

if(MSVC)
  target_compile_options(nags_basic PRIVATE /W4 /permissive-)
  target_compile_options(nags_fast PRIVATE /W4 /permissive-)
  target_compile_options(nags_enhanced PRIVATE /W4 /permissive-)
  target_compile_options(nags PRIVATE /W4 /permissive-)
else()
  target_compile_options(nags_basic PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(nags_fast PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(nags_enhanced PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(nags PRIVATE -Wall -Wextra -Wpedantic)
endif()


