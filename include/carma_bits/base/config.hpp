#pragma once

#include <carma_bits/internal/converter_types.hpp>
#ifndef CARMA_DEFAULT_MEMORY_ORDER
#define CARMA_DEFAULT_MEMORY_ORDER carma::ColumnOrder
#endif  // CARMA_DEFAULT_MEMORY_ORDER

#ifdef CARMA_EXTRA_DEBUG
#include <carma_bits/base/converter_types.hpp>
#include <carma_bits/version.hpp>
#include <iostream>

namespace carma {
namespace anon {
class carma_config_debug_message {
   public:
    inline carma_config_debug_message() {
        std::cout << "\n|----------------------------------------------------------|\n"
                  << "|                    CARMA CONFIGURATION                   |"
                  << "\n|----------------------------------------------------------|\n|\n";
        std::cout << "| Carma version: " + carma_version().as_string() << "\n";
        std::cout << "| Carma mode: base\n|\n";
        std::cout << "| Default Numpy to Arma conversion config:\n"
                  << "| ----------------------------------------\n"
                  << "| * l-value converter:                 " << CopyConverter::name_ << "\n"
                  << "| * const l-value converter:           " << CopyConverter::name_ << "\n"
                  << "| * memory_order_policy:               " << CARMA_DEFAULT_MEMORY_ORDER::name_ << "\n";
        std::cout << "|\n|----------------------------------------------------------|\n\n";
    };
};

static const carma_config_debug_message carma_config_debug_message_print;
}  // namespace anon
}  // namespace carma
#endif  // CARMA_EXTRA_DEBUG
