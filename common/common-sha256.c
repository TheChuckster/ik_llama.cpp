/*
 * Public-domain SHA-256 implementation derived from Igor Pavlov's 2010 code,
 * itself based on Wei Dai's public-domain Crypto++ implementation.
 */

#include "common-sha256.h"

#define ROTR32(value, count) (((value) >> (count)) | ((value) << (32 - (count))))
#define BIG_SIGMA_0(value) (ROTR32(value, 2) ^ ROTR32(value, 13) ^ ROTR32(value, 22))
#define BIG_SIGMA_1(value) (ROTR32(value, 6) ^ ROTR32(value, 11) ^ ROTR32(value, 25))
#define SMALL_SIGMA_0(value) (ROTR32(value, 7) ^ ROTR32(value, 18) ^ ((value) >> 3))
#define SMALL_SIGMA_1(value) (ROTR32(value, 17) ^ ROTR32(value, 19) ^ ((value) >> 10))
#define CHOOSE(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define MAJORITY(x, y, z) (((x) & (y)) | ((z) & ((x) | (y))))

static const uint32_t common_sha256_constants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static void common_sha256_transform(uint32_t state[8], const uint32_t input[16]) {
    uint32_t schedule[64];
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (size_t index = 0; index < 16; ++index) {
        schedule[index] = input[index];
    }
    for (size_t index = 16; index < 64; ++index) {
        schedule[index] = SMALL_SIGMA_1(schedule[index - 2])
            + schedule[index - 7]
            + SMALL_SIGMA_0(schedule[index - 15])
            + schedule[index - 16];
    }
    for (size_t index = 0; index < 64; ++index) {
        const uint32_t first = h + BIG_SIGMA_1(e) + CHOOSE(e, f, g)
            + common_sha256_constants[index] + schedule[index];
        const uint32_t second = BIG_SIGMA_0(a) + MAJORITY(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

static void common_sha256_write_block(common_sha256_context * context) {
    uint32_t words[16];
    for (size_t index = 0; index < 16; ++index) {
        words[index] = ((uint32_t) context->buffer[index * 4] << 24)
            | ((uint32_t) context->buffer[index * 4 + 1] << 16)
            | ((uint32_t) context->buffer[index * 4 + 2] << 8)
            | ((uint32_t) context->buffer[index * 4 + 3]);
    }
    common_sha256_transform(context->state, words);
}

void common_sha256_init(common_sha256_context * context) {
    context->state[0] = 0x6a09e667;
    context->state[1] = 0xbb67ae85;
    context->state[2] = 0x3c6ef372;
    context->state[3] = 0xa54ff53a;
    context->state[4] = 0x510e527f;
    context->state[5] = 0x9b05688c;
    context->state[6] = 0x1f83d9ab;
    context->state[7] = 0x5be0cd19;
    context->count = 0;
}

void common_sha256_update(
        common_sha256_context * context,
        const unsigned char * data,
        size_t size) {
    uint32_t position = (uint32_t) context->count & 0x3f;
    while (size > 0) {
        context->buffer[position++] = *data++;
        ++context->count;
        --size;
        if (position == 64) {
            position = 0;
            common_sha256_write_block(context);
        }
    }
}

void common_sha256_final(
        common_sha256_context * context,
        unsigned char digest[COMMON_SHA256_DIGEST_SIZE]) {
    uint64_t length_bits = context->count << 3;
    uint32_t position = (uint32_t) context->count & 0x3f;
    context->buffer[position++] = 0x80;
    while (position != 56) {
        position &= 0x3f;
        if (position == 0) {
            common_sha256_write_block(context);
        }
        context->buffer[position++] = 0;
    }
    for (size_t index = 0; index < 8; ++index) {
        context->buffer[position++] = (unsigned char) (length_bits >> 56);
        length_bits <<= 8;
    }
    common_sha256_write_block(context);

    for (size_t index = 0; index < 8; ++index) {
        *digest++ = (unsigned char) (context->state[index] >> 24);
        *digest++ = (unsigned char) (context->state[index] >> 16);
        *digest++ = (unsigned char) (context->state[index] >> 8);
        *digest++ = (unsigned char) context->state[index];
    }
    common_sha256_init(context);
}

void common_sha256_hash(
        unsigned char digest[COMMON_SHA256_DIGEST_SIZE],
        const unsigned char * data,
        size_t size) {
    common_sha256_context context;
    common_sha256_init(&context);
    common_sha256_update(&context, data, size);
    common_sha256_final(&context, digest);
}

#undef ROTR32
#undef BIG_SIGMA_0
#undef BIG_SIGMA_1
#undef SMALL_SIGMA_0
#undef SMALL_SIGMA_1
#undef CHOOSE
#undef MAJORITY
