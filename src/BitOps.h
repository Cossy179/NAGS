#pragma once

// Portable bit-twiddling helpers. MSVC, GCC and Clang all get native
// instructions; anything else falls back to plain C++.

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// Index of the least significant set bit. Undefined for b == 0.
inline int lsb(uint64_t b) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(b);
#else
    int i = 0;
    while (!(b & 1ULL)) { b >>= 1; ++i; }
    return i;
#endif
}

// Index of the most significant set bit. Undefined for b == 0.
inline int msb(uint64_t b) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return static_cast<int>(idx);
#elif defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(b);
#else
    int i = 63;
    while (!(b & (1ULL << 63))) { b <<= 1; --i; }
    return i;
#endif
}

inline int popcount(uint64_t b) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(b));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#else
    int n = 0;
    while (b) { b &= b - 1; ++n; }
    return n;
#endif
}

// Removes and returns the least significant set bit. Undefined for b == 0.
inline int popLsb(uint64_t &b) {
    int sq = lsb(b);
    b &= b - 1;
    return sq;
}
