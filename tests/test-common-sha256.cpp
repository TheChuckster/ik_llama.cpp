#include "common-sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static std::string hexadecimal_digest(
        const unsigned char digest[COMMON_SHA256_DIGEST_SIZE]) {
    static const char hexadecimal[] = "0123456789abcdef";
    std::string result(COMMON_SHA256_DIGEST_SIZE * 2, '0');
    for (size_t index = 0; index < COMMON_SHA256_DIGEST_SIZE; ++index) {
        result[index * 2] = hexadecimal[digest[index] >> 4];
        result[index * 2 + 1] = hexadecimal[digest[index] & 0x0f];
    }
    return result;
}

static void check_one_shot(const char * input, size_t size, const char * expected) {
    unsigned char digest[COMMON_SHA256_DIGEST_SIZE];
    common_sha256_hash(
        digest, reinterpret_cast<const unsigned char *>(input), size);
    CHECK(hexadecimal_digest(digest) == expected);
}

int main() {
    check_one_shot(nullptr, 0,
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855");
    check_one_shot("abc", 3,
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad");
    static const char multi_block[] =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check_one_shot(multi_block, std::strlen(multi_block),
        "248d6a61d20638b8e5c026930c3e6039"
        "a33ce45964ff2167f6ecedd419db06c1");

    common_sha256_context context;
    common_sha256_init(&context);
    const std::vector<unsigned char> thousand(1000, 'a');
    for (int index = 0; index < 1000; ++index) {
        common_sha256_update(&context, thousand.data(), thousand.size());
    }
    unsigned char digest[COMMON_SHA256_DIGEST_SIZE];
    common_sha256_final(&context, digest);
    CHECK(hexadecimal_digest(digest) ==
        "cdc76e5c9914fb9281a1c7e284d73e67"
        "f1809a48a497200e046d39ccc7112cd0");

    std::puts("common SHA-256 test passed");
    return 0;
}
