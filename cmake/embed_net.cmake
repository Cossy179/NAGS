# Writes OUTPUT, a C++ source holding the bytes of INPUT (the default NNUE
# network) as kNagsEmbeddedNet. Without INPUT the array is empty and the
# engines use the hand-written evaluation unless a network is loaded.
if(INPUT AND EXISTS "${INPUT}")
  file(READ "${INPUT}" hex HEX)
  file(SIZE "${INPUT}" size)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
  string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n" bytes "${bytes}")
else()
  set(size 0)
  set(bytes "0")
endif()
file(WRITE "${OUTPUT}.tmp"
  "// Generated from ${INPUT} by cmake/embed_net.cmake.\n"
  "extern const unsigned char kNagsEmbeddedNet[] = {\n${bytes}\n};\n"
  "extern const unsigned long kNagsEmbeddedNetSize = ${size}UL;\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${OUTPUT}.tmp" "${OUTPUT}")
file(REMOVE "${OUTPUT}.tmp")
