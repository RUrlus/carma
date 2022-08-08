#ifndef INCLUDE_CARMA_
#define INCLUDE_CARMA_

#include <carma_bits/to_arma.hpp>
#include <iostream>

namespace carma {

#if defined(CARMA_EXTRA_DEBUG)

#if not defined(CARMA_DEBUG)
#define CARMA_DEBUG true
#endif  // CARMA_DEBUG

namespace junk {
class carma_config_debug_message {
   public:
    inline carma_config_debug_message() {
        std::cout << "\n|----------------------------------------------------------|\n"
                  << "|                    CARMA CONFIGURATION                   |"
                  << "\n|----------------------------------------------------------|\n|\n";
        std::cout << "| Default Numpy to Arma conversion config:\n"
                  << "| ----------------------------------------\n"
                  << "| * l-value converter = " << CARMA_DEFAULT_LVALUE_CONVERTER::name_ << "\n"
                  << "| * const l-value converter = " << CARMA_DEFAULT_CONST_LVALUE_CONVERTER::name_ << "\n"
                  << "| * resolution_policy = " << CARMA_DEFAULT_RESOLUTION::name_ << "\n"
                  << "| * memory_order_policy = " << CARMA_DEFAULT_MEMORY_ORDER::name_ << "\n";
        std::cout << "|\n|----------------------------------------------------------|\n\n";
    };
};

static carma_config_debug_message carma_config_debug_message_print;
}  // namespace junk
#endif  // CARMA_EXTRA_DEBUG

}  // namespace carma

#endif  // INCLUDE_CARMA_
