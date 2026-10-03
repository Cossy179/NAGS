# Runs ENGINE with INPUT on stdin and checks that the output matches EXPECT
# (a regular expression; newlines are replaced by spaces first).
if(NOT EXISTS "${INPUT}")
  message(FATAL_ERROR "input file ${INPUT} not found")
endif()
execute_process(
  COMMAND "${ENGINE}"
  INPUT_FILE "${INPUT}"
  OUTPUT_VARIABLE out
  ERROR_VARIABLE err
  RESULT_VARIABLE rc
  TIMEOUT 120)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "engine exited with '${rc}'\n${out}\n${err}")
endif()
string(REPLACE "\r" "" out "${out}")
string(REPLACE "\n" " " flat "${out}")
if(NOT flat MATCHES "${EXPECT}")
  message(FATAL_ERROR "output did not match '${EXPECT}':\n${out}")
endif()
message(STATUS "ok: ${EXPECT}")
