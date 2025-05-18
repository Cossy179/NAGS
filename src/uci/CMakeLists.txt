set(UCI_SOURCES
    uci.cpp
)

set(UCI_HEADERS
    uci.h
)

add_library(uci STATIC ${UCI_SOURCES} ${UCI_HEADERS})
target_include_directories(uci PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(uci PRIVATE core) 