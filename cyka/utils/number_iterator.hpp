//
// Created by Joseph on 2024/5/9.
//

#ifndef CYKA_NUMBER_ITERATOR_HPP
#define CYKA_NUMBER_ITERATOR_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace cyka::utils {

/// Wrap an integer type into iterator
template <typename T>
  requires std::is_arithmetic_v<T>
struct number_iterator {
  T value{0};

  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using difference_type = T;
  using pointer = T *;
  using reference = T &;

  T &operator*() noexcept { return this->value; }
  number_iterator &operator++() noexcept {
    this->value++;
    return *this;
  }

  number_iterator operator++(int) noexcept {
    number_iterator ret{*this};
    this->value++;
    return ret;
  }

  [[nodiscard]] bool operator!=(number_iterator b) const noexcept {
    return this->value not_eq b.value;
  }

  [[nodiscard]] bool operator==(number_iterator b) const noexcept {
    return not this->operator!=(b);
  }
};
} // namespace cyka::utils

#endif // CYKA_NUMBER_ITERATOR_HPP
