#pragma once

#include <carma_bits/extension/converter_types.hpp>
/* --------------------------------------------------------------
                    ConversionConfig
-------------------------------------------------------------- */
#ifndef CARMA_DEFAULT_LVALUE_CONVERTER
#define CARMA_DEFAULT_LVALUE_CONVERTER carma::BorrowConverter
#endif  // CARMA_DEFAULT_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_CONST_LVALUE_CONVERTER
#define CARMA_DEFAULT_CONST_LVALUE_CONVERTER carma::CopyConverter
#endif  // CARMA_DEFAULT_CONST_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_RESOLUTION
#define CARMA_DEFAULT_RESOLUTION carma::CopyResolution
#endif  // CARMA_DEFAULT_RESOLUTION

#ifndef CARMA_DEFAULT_MEMORY_ORDER
#define CARMA_DEFAULT_MEMORY_ORDER carma::ColumnOrder
#endif  // CARMA_DEFAULT_MEMORY_ORDER
// converters

#ifdef CARMA_EXTRA_DEBUG

#include <carma_bits/version.hpp>
#include <iostream>
namespace carma::anon {
class carma_config_debug_message {
   public:
    inline carma_config_debug_message() {
        std::cout << "\n|----------------------------------------------------------|\n"
                  << "|                    CARMA CONFIGURATION                   |"
                  << "\n|----------------------------------------------------------|\n|\n";
        std::cout << "| Carma version: " + carma_version().as_string() << "\n";
        std::cout << "| Carma mode: extension\n|\n";
        std::cout << "| Default Numpy to Arma conversion config:\n"
                  << "| ----------------------------------------\n"
                  << "| * l-value converter:                 " << CARMA_DEFAULT_LVALUE_CONVERTER::name_ << "\n"
                  << "| * const l-value converter:           " << CARMA_DEFAULT_CONST_LVALUE_CONVERTER::name_ << "\n"
                  << "| * resolution_policy:                 " << CARMA_DEFAULT_RESOLUTION::name_ << "\n"
                  << "| * memory_order_policy:               " << CARMA_DEFAULT_MEMORY_ORDER::name_ << "\n";
        std::cout << "|\n| Converter Options:\n"
                  << "| ------------------\n"
                  << "| * enforce rvalue for MoveConverter:  "
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
                  << "true\n";
#else
                  << "false\n";
#endif  // CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        std::cout << "|\n|----------------------------------------------------------|\n\n";
    };
};

static const carma_config_debug_message carma_config_debug_message_print;
}  // namespace carma::anon
#endif  // CARMA_EXTRA_DEBUG
