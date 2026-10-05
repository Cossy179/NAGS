# Runs "ENGINE bench DEPTH" twice and checks that the node count is identical
# (the bench fingerprint must be deterministic).
foreach(run 1 2)
  execute_process(
    COMMAND "${ENGINE}" bench ${DEPTH}
    OUTPUT_VARIABLE out
    RESULT_VARIABLE rc
    TIMEOUT 300)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "bench exited with '${rc}'\n${out}")
  endif()
  string(REGEX MATCH "Nodes searched  : ([0-9]+)" match "${out}")
  if(NOT match)
    message(FATAL_ERROR "no 'Nodes searched' line in bench output:\n${out}")
  endif()
  set(nodes_${run} "${CMAKE_MATCH_1}")
endforeach()
if(NOT nodes_1 STREQUAL nodes_2)
  message(FATAL_ERROR "bench is not deterministic: ${nodes_1} vs ${nodes_2} nodes")
endif()
message(STATUS "bench ${DEPTH}: ${nodes_1} nodes (deterministic)")
