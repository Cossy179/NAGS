set(CORE_SOURCES
    engine.cpp
    board.cpp
    search.cpp
    evaluator.cpp
)

set(CORE_HEADERS
    engine.h
    board.h
    search.h
    bitboard.h
    search_options.h
    evaluator.h
)

add_library(core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PRIVATE Threads::Threads) 