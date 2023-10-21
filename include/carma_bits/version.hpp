#pragma once
#include <sstream>
#include <string>

#ifndef CARMA_VERSION_MAJOR
#define CARMA_VERSION_MAJOR 1
#define CARMA_VERSION_MINOR 0
#define CARMA_VERSION_PATCH 0
#define CARMA_VERSION_NAME "1.0.0 HO"
#endif

namespace carma {

struct carma_version {
    static constexpr unsigned int major = CARMA_VERSION_MAJOR;
    static constexpr unsigned int minor = CARMA_VERSION_MINOR;
    static constexpr unsigned int patch = CARMA_VERSION_PATCH;

    static inline std::string as_string() {
        std::ostringstream buffer;
        buffer << carma_version::major << "." << carma_version::minor << "." << carma_version::patch;
        return buffer.str();
    }
};  // carma_version

}  // namespace carma
