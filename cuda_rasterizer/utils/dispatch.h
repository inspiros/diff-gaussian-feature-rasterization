#pragma once

#include <string>
#include <ATen/Dispatch.h>
#include "../config.h"
#include "map.h"

// TODO: num_channels must be known at compile time in this list.
//  Each predefined num_channels generate a dedicated CUDA kernel,
//  thus, only add what you think would be used.
#define _STRINGIZE(...) #__VA_ARGS__
#define STRINGIZE(...) _STRINGIZE(__VA_ARGS__)
#define DGRF_ERROR_MSG "\nPre-defined channels are [" STRINGIZE(DGRF_PREDEFINED_CHANNELS) "]." \
    "\nDue to a limitation of CUDA, num_channels must be known at compile time. " \
    "\nPlease add your desired dimensionality to DGRF_PREDEFINED_CHANNELS " \
    "in 'cuda_rasterizer/config.h' and recompile."

#define DGRF_GET_MAX_CHANNELS(BS) ((48 * 1024 - BS * 28) / (BS * 4))

// TODO: There is a technical limit of 48kbs shared memory in CUDA kernels,
//  too large dimensionality will cause ptxas error in BACKWARD::render.
//  DGRF_DISPATCH_CASE tries to select the best dynamic block size that fits the data
template<uint32_t C>
static constexpr uint32_t get_block_resolution() {
    if constexpr (C <= DGRF_GET_MAX_CHANNELS(16 * 16))
        return 16;
    else if constexpr (C <= DGRF_GET_MAX_CHANNELS(8 * 8))
        return 8;
    else if constexpr (C <= DGRF_GET_MAX_CHANNELS(4 * 4))
        return 4;
    else if constexpr (C <= DGRF_GET_MAX_CHANNELS(2 * 2))
        return 2;
    else if constexpr (C <= DGRF_GET_MAX_CHANNELS(1 * 1))
        return 1;
    else
        static_assert(false, "too large channels");
}

template<uint32_t C>
static constexpr uint32_t get_block_size() {
    return get_block_resolution<C>() * get_block_resolution<C>();
}

#define DGRF_DISPATCH_CASE(C, ...)              \
  case (C): {                                   \
    [[maybe_unused]] static constexpr uint32_t num_channels = C; \
    [[maybe_unused]] static constexpr auto block_resolution = get_block_resolution<num_channels>(); \
    [[maybe_unused]] static constexpr auto block_size = block_resolution * block_resolution; \
    return __VA_ARGS__();                       \
  }

#define DGRF_DISPATCH_SWITCH(C, NAME, ...)           \
  [&] {                                              \
    constexpr const char* dgrf_dispatch_name = NAME; \
    switch (C) {                                     \
      __VA_ARGS__                                    \
      default:                                       \
        TORCH_CHECK(                                 \
            false,                                   \
            '"',                                     \
            dgrf_dispatch_name,                      \
            "\" not implemented for num_channels=",  \
            std::to_string(C),                       \
            ". ",                                    \
            DGRF_ERROR_MSG);                         \
    }                                                \
  }()

#define DGRF_DISPATCH_PREDEFINED_CHANNELS(C, NAME, ...) \
    DGRF_DISPATCH_SWITCH(C, NAME, MAP_UD(DGRF_DISPATCH_CASE, __VA_ARGS__, DGRF_PREDEFINED_CHANNELS))
