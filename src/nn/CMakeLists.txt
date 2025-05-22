# Neural Network Module
add_library(nn
    gnn_evaluator.cpp
)

target_include_directories(nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nn PRIVATE Threads::Threads) 