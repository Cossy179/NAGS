set(META_SOURCES
    meta_learner.cpp
    bayesian_bandit.cpp
)

set(META_HEADERS
    meta_learner.h
    bayesian_bandit.h
)

add_library(meta STATIC ${META_SOURCES} ${META_HEADERS})
target_include_directories(meta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(meta PRIVATE core Threads::Threads) 