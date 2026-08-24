#include "cvector-layer-capture.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        std::abort(); \
    } \
} while (0)

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

    std::puts("cvector layer capture tests passed");
    return 0;
}
