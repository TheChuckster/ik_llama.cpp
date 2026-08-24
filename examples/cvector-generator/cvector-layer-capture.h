#ifndef CVECTOR_LAYER_CAPTURE_H
#define CVECTOR_LAYER_CAPTURE_H

#include <climits>
#include <cstring>

inline int cvector_l_out_layer_index(const char * name) {
    static constexpr char prefix[] = "l_out-";
    if (std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return -1;
    }

    const char * cursor = name + sizeof(prefix) - 1;
    if (*cursor == '\0') {
        return -1;
    }
    int result = 0;
    while (*cursor != '\0') {
        if (*cursor < '0' || *cursor > '9') {
            return -1;
        }
        const int digit = *cursor - '0';
        if (result > (INT_MAX - digit) / 10) {
            return -1;
        }
        result = result * 10 + digit;
        ++cursor;
    }
    return result;
}

inline bool cvector_should_capture_layer(const char * name, int n_layers) {
    const int layer = cvector_l_out_layer_index(name);
    return n_layers > 1 && layer >= 0 && layer < n_layers - 1;
}

#endif // CVECTOR_LAYER_CAPTURE_H
