#pragma once

// The 'relational-v1' graph of a position (relational_graph.py), built in the
// engine so that the GNN server does not have to: the same 32 features, the
// same 21 relations, the same nodes and the same edges in the same order as
// relational_graph.build_graph for the position's FEN (tests/python/
// test_relgraph.py compares them on thousands of positions). Like a FEN, it
// carries no game history: the 'repetitions' feature is 0.
//
// encodeGraph packs it with the legal moves for the RPC protocol ("graphs",
// see rpc_server.py and relational_fast.decode_graph), little endian, base64:
// uint8 nodes, uint16 edges, uint16 nonzero features, uint8 legal moves; the
// nonzero features' indices (uint16 node * 32 + column), then their values
// (float32); the edges sorted by destination (uint8 source, uint8
// destination, uint8 relation); the legal moves' v3 policy indices (uint16).

#include "FastBoard.h"

#include <cstdint>
#include <string>
#include <vector>

struct RelGraph {
    static constexpr int kFeatures = 32;
    int nodes = 0;
    std::vector<float> x;                     // nodes * kFeatures
    std::vector<uint8_t> src, dst, relation;  // sorted by dst (stable)
    std::vector<uint16_t> legal;              // v3 policy index of each legal move
};

// policy_encoding.move_to_index_v3: from * 64 + to, underpromotions after 4096.
int policyIndexV3(const Move &m);

void buildRelationalGraph(const FastBoard &board, RelGraph &out);
std::string encodeGraph(const RelGraph &g);
