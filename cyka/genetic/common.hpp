//
// Created by Joseph on 2024/5/8.
//

#ifndef CYKA_COMMON_HPP
#define CYKA_COMMON_HPP

#include <cstddef>
#include <cstdint>

#if !defined(__cpp_size_t_suffix) || __cpp_size_t_suffix < 202011L
inline constexpr std::size_t operator"" uz(unsigned long long const value) {
  return {value};
}
inline constexpr std::size_t operator"" zu(unsigned long long const value) {
  return {value};
}
#endif

namespace cyka {

inline constexpr std::ptrdiff_t
operator"" _isize(unsigned long long const val) {
  return static_cast<std::ptrdiff_t>(val);
}
} // namespace cyka

#endif // CYKA_COMMON_HPP
