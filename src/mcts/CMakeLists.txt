# Monte Carlo Tree Search Module
add_library(mcts
    mcts.cpp
)

target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts PRIVATE Threads::Threads) 