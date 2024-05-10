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

#include "loss_computer.hpp"

namespace cyka::genetic {

template <size_t n_obj, class option_t> class selector_base {
public:
  using loss_type = loss_value_type<n_obj>;
  using loss_matrix = loss_matrix_type<n_obj>;
  static constexpr size_t num_objectives_template_parameter = n_obj;

  virtual ~selector_base() = default;

  virtual void select(const loss_matrix &loss_of_whole_group,
                      size_t expected_group_size,
                      Eigen::ArrayX<uint16_t> &select_count,
                      std::mt19937 &rand_engine) noexcept = 0;

  [[nodiscard]] Eigen::ArrayX<uint16_t>
  select(const loss_matrix &loss_of_whole_group,
         size_t expected_group_size, std::mt19937 &rand_engine) noexcept {
    Eigen::ArrayX<uint16_t> dest;
    this->select(loss_of_whole_group, expected_group_size, dest,
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

template <class selector_t>
concept is_selector =
    std::is_base_of_v<
        selector_base<selector_t::num_objectives_template_parameter,
                      typename selector_t::select_option_type>,
        selector_t> and
    requires(const selector_t &selector_const, selector_t &selector_mut,
             const selector_t::loss_matrix &fm,
             Eigen::ArrayX<uint16_t> &select_count, std::mt19937 &rand_engine,
             selector_t::select_option_type option) {
      selector_mut.select(fm, 0zu, select_count, rand_engine);
      selector_const.select_option();
      selector_const.check_select_option(option);
      selector_mut.set_select_option(std::move(option));
    };

} // namespace cyka::genetic

#endif // CYKA_SELECTOR_HPP
