# Runs "bench DEPTH" on ENGINE and ENGINE2 and checks that the node counts
# are equal (nags without its Python services must search exactly like
# nags_enhanced).
foreach(engine ENGINE ENGINE2)
  execute_process(
    COMMAND "${${engine}}" bench ${DEPTH}
    OUTPUT_VARIABLE out
    RESULT_VARIABLE rc
    TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${${engine}} bench exited with '${rc}'\n${out}")
  endif()
  string(REGEX MATCH "Nodes searched  : ([0-9]+)" match "${out}")
  if(NOT match)
    message(FATAL_ERROR "no 'Nodes searched' line in bench output:\n${out}")
  endif()
  set(nodes_${engine} "${CMAKE_MATCH_1}")
endforeach()
if(NOT nodes_ENGINE STREQUAL nodes_ENGINE2)
  message(FATAL_ERROR "bench differs: ${nodes_ENGINE} vs ${nodes_ENGINE2} nodes")
endif()
message(STATUS "bench ${DEPTH}: ${nodes_ENGINE} nodes on both engines")
