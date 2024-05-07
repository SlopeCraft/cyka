//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_SELECTOR_HPP
#define CYKA_SELECTOR_HPP

#include <exception>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include "fitness_computer.hpp"

namespace cyka::genetic {

template <size_t n_obj, class option_t> class selector_base {
public:
  using fitness_type = fitness_value_type<n_obj>;
  using fitness_matrix = fitness_matrix_type<n_obj>;

  virtual ~selector_base() = default;

  virtual void select(const fitness_matrix &fitness_of_whole_group,
                      size_t expected_group_size,
                      Eigen::ArrayX<uint16_t> &select_count,
                      std::mt19937 &rand_engine) const noexcept = 0;

  [[nodiscard]] Eigen::ArrayX<uint16_t>
  select(const fitness_matrix &fitness_of_whole_group,
         size_t expected_group_size, std::mt19937 &rand_engine) const noexcept {
    Eigen::ArrayX<uint16_t> dest;
    this->select(fitness_of_whole_group, expected_group_size, dest,
                 rand_engine);
    return dest;
  }

  using select_option_type = option_t;

protected:
  select_option_type select_option_;

public:
  [[nodiscard]] const select_option_type &select_option() const noexcept {
    return this->select_option_;
  }

  [[nodiscard]] virtual std::optional<std::invalid_argument>
  check_select_option(const select_option_type &) const noexcept {
    return std::nullopt;
  };

  void set_select_option(select_option_type &&opt) {
    auto result = this->check_select_option(opt);
    if (result) {
      throw(std::move(result.value()));
    }
    this->select_option_ = opt;
  }
};

} // namespace cyka::genetic

#endif // CYKA_SELECTOR_HPP
