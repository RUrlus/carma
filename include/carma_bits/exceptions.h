#ifndef INCLUDE_CARMA_BITS_EXCEPTIONS_H_
#define INCLUDE_CARMA_BITS_EXCEPTIONS_H_
#include <stdexcept>

namespace carma {

class ConversionError : public std::runtime_error {
 public:
    explicit ConversionError(const char* what) : std::runtime_error(what) {}
};

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_EXCEPTIONS_H_
