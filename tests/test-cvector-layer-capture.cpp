#include "cvector-layer-capture.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

static bool parse_throws(const std::string & spec, int max_layer) {
    try {
        (void) cvector_parse_layer_spec(spec, max_layer);
        return false;
    } catch (const std::invalid_argument &) {
        return true;
    }
}

int main() {
    CHECK(cvector_l_out_layer_index("l_out-0") == 0);
    CHECK(cvector_l_out_layer_index("l_out-91") == 91);
    CHECK(cvector_l_out_layer_index("l_out-2147483647") == 2147483647);

    CHECK(cvector_l_out_layer_index("l_out") == -1);
    CHECK(cvector_l_out_layer_index("l_out-") == -1);
    CHECK(cvector_l_out_layer_index("l_out--1") == -1);
    CHECK(cvector_l_out_layer_index("l_out-1-extra") == -1);
    CHECK(cvector_l_out_layer_index("l_out_selected-1") == -1);
    CHECK(cvector_l_out_layer_index("l_out-2147483648") == -1);

    CHECK(cvector_should_capture_layer("l_out-0", 93));
    CHECK(cvector_should_capture_layer("l_out-91", 93));
    CHECK(!cvector_should_capture_layer("l_out-92", 93));
    CHECK(!cvector_should_capture_layer("l_out-93", 93));
    CHECK(!cvector_should_capture_layer("l_out-0", 1));
    CHECK(!cvector_should_capture_layer("l_out_selected-1", 93));

    CHECK(cvector_parse_layer_spec("56-58,64,58", 92) == std::vector<int>({56, 57, 58, 64}));
    CHECK(cvector_parse_layer_spec("1,92", 92) == std::vector<int>({1, 92}));
    CHECK(parse_throws("", 92));
    CHECK(parse_throws("0", 92));
    CHECK(parse_throws("3-2", 92));
    CHECK(parse_throws("93", 92));
    CHECK(parse_throws("1, 2", 92));
    CHECK(parse_throws("1,,2", 92));

    std::puts("cvector layer capture tests passed");
    return 0;
}
