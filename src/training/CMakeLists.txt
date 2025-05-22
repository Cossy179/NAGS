set(TRAINING_SOURCES
    supervised_trainer.cpp
    rl_trainer.cpp
    dual_training_pipeline.cpp
)

set(TRAINING_HEADERS
    supervised_trainer.h
    rl_trainer.h
    dual_training_pipeline.h
)

add_library(training STATIC ${TRAINING_SOURCES} ${TRAINING_HEADERS})
target_link_libraries(training PUBLIC core meta nn gnn mcts) 