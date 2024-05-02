//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_SELECTOR_HPP
#define CYKA_SELECTOR_HPP

#include <random>
#include <type_traits>
#include <vector>

#include "fitness_computer.hpp"

namespace cyka::genetic {

template <size_t n_obj> class selector_base {
public:
  using fitness_type = fitness_computer<n_obj>::fitness_type;
  using fitness_matrix = fitness_computer<n_obj>::fitness_matrix;

  virtual ~selector_base() = default;

  virtual void select(const fitness_matrix &fitness_of_whole_group,
                      size_t expected_group_size, Eigen::ArrayX<bool> &dest,
                      std::mt19937 &rand_engine) noexcept = 0;

  [[nodiscard]] Eigen::ArrayX<bool>
  select(const fitness_matrix &fitness_of_whole_group,
         size_t expected_group_size, std::mt19937 &rand_engine) noexcept {
    Eigen::ArrayX<bool> dest;
    this->select(fitness_of_whole_group, expected_group_size, dest,
                 rand_engine);
    return dest;
  }
};

} // namespace cyka::genetic

#endif // CYKA_SELECTOR_HPP
