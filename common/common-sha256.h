/* Public-domain SHA-256 interface derived from Igor Pavlov's 2010 implementation. */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COMMON_SHA256_DIGEST_SIZE 32

typedef struct common_sha256_context {
    uint32_t state[8];
    uint64_t count;
    unsigned char buffer[64];
} common_sha256_context;

void common_sha256_init(common_sha256_context * context);
void common_sha256_update(
        common_sha256_context * context,
        const unsigned char * data,
        size_t size);
void common_sha256_final(
        common_sha256_context * context,
        unsigned char digest[COMMON_SHA256_DIGEST_SIZE]);
void common_sha256_hash(
        unsigned char digest[COMMON_SHA256_DIGEST_SIZE],
        const unsigned char * data,
        size_t size);

#ifdef __cplusplus
}
#endif
