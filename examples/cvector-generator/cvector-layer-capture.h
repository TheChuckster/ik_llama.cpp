#ifndef CVECTOR_LAYER_CAPTURE_H
#define CVECTOR_LAYER_CAPTURE_H

#include <climits>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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

inline int cvector_parse_positive_layer(const std::string & value) {
    if (value.empty()) {
        throw std::invalid_argument("empty layer number");
    }
    int result = 0;
    for (const char ch : value) {
        if (ch < '0' || ch > '9') {
            throw std::invalid_argument("layer numbers must contain only decimal digits");
        }
        const int digit = ch - '0';
        if (result > (INT_MAX - digit) / 10) {
            throw std::invalid_argument("layer number overflows int");
        }
        result = result * 10 + digit;
    }
    if (result < 1) {
        throw std::invalid_argument("layer numbers are 1-based");
    }
    return result;
}

// Parse a deliberately small, deterministic 1-based layer grammar:
//     SPEC := ITEM (',' ITEM)*
//     ITEM := N | N '-' N
// Duplicates are collapsed and the result is sorted. Whitespace and descending
// ranges are rejected so a recorded command identifies one canonical set.
inline std::vector<int> cvector_parse_layer_spec(const std::string & spec, int max_layer) {
    if (spec.empty()) {
        throw std::invalid_argument("activation layer specification is empty");
    }
    if (max_layer < 1) {
        throw std::invalid_argument("model has no capturable layers");
    }

    std::set<int> selected;
    size_t begin = 0;
    while (begin <= spec.size()) {
        const size_t comma = spec.find(',', begin);
        const std::string item = spec.substr(
                begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (item.empty()) {
            throw std::invalid_argument("activation layer specification contains an empty item");
        }
        const size_t dash = item.find('-');
        if (dash != std::string::npos && item.find('-', dash + 1) != std::string::npos) {
            throw std::invalid_argument("activation layer range contains more than one dash");
        }
        const int first = cvector_parse_positive_layer(item.substr(0, dash));
        const int last = dash == std::string::npos
                ? first
                : cvector_parse_positive_layer(item.substr(dash + 1));
        if (last < first) {
            throw std::invalid_argument("activation layer ranges must be ascending");
        }
        if (last > max_layer) {
            throw std::invalid_argument("activation layer is outside the model");
        }
        for (int layer = first; layer <= last; ++layer) {
            selected.insert(layer);
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return std::vector<int>(selected.begin(), selected.end());
}

#endif // CVECTOR_LAYER_CAPTURE_H
