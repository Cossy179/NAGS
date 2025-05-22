# Graph Neural Network Module (Python files)
# The GNN implementation is in Python, so we create an empty library
# The Python files will be used by the training scripts

add_library(gnn INTERFACE)
target_include_directories(gnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}) 