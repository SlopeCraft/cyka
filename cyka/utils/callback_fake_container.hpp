//
// Created by Joseph on 2024/5/9.
//

#ifndef CYKA_CALLBACK_FAKE_CONTAINER_HPP
#define CYKA_CALLBACK_FAKE_CONTAINER_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace cyka::utils {

/// Wrap a callback into a fake-container, the push_back method will call the
/// callback. Designed to use with std::back_inserter to wrap a callback into an
/// iterator
template <class callback_t, class T>
  requires std::is_invocable_r_v<void, const callback_t, T>
struct callback_fake_container {
  callback_fake_container() = delete;
  explicit callback_fake_container(const callback_t *f) : fun{f} {}
  explicit callback_fake_container(const callback_t &f) : fun{&f} {}

  const callback_t *fun;

  void push_back(T &&t) { (*this->fun)(t); }
  void push_back(const T &t) { (*this->fun)(t); }

  using value_type = T;
};

} // namespace cyka::utils

#endif // CYKA_CALLBACK_FAKE_CONTAINER_HPP
