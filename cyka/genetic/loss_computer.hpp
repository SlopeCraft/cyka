//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_LOSS_COMPUTER_HPP
#define CYKA_LOSS_COMPUTER_HPP

#include <cstddef>
#include <cstdint>

#include <Eigen/Dense>

#include "population_base.hpp"

namespace cyka::genetic {

namespace detail {

// fixed multiple objectives
template <size_t n_objectives> class loss_computer_impl {
public:
  using loss_type = Eigen::Array<double, n_objectives, 1>;

  virtual ~loss_computer_impl() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept {
    return n_objectives;
  }
};

template <> class loss_computer_impl<1> {
public:
  using loss_type = double;

  virtual ~loss_computer_impl() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept { return 1; }
};

template <> class loss_computer_impl<0> {
public:
  using loss_type = Eigen::ArrayXd;

  virtual ~loss_computer_impl() = default;

  [[nodiscard]] virtual size_t num_objectives() const noexcept = 0;
};
} // namespace detail

template <size_t n_obj>
using loss_value_type = detail::loss_computer_impl<n_obj>::loss_type;

template <size_t n_obj>
using loss_matrix_type =
    Eigen::Array<double, n_obj == 0 ? Eigen::Dynamic : n_obj, Eigen::Dynamic>;

template <class const_gene_view, class loss>
using loss_function = loss(const_gene_view);

template <size_t n_obj, class const_gene_view>
class loss_computer : public detail::loss_computer_impl<n_obj> {
public:
  using base_t = detail::loss_computer_impl<n_obj>;
  using loss_type = base_t::loss_type;
  using loss_matrix_type = loss_matrix_type<n_obj>;

  virtual ~loss_computer() = default;

  [[nodiscard]] virtual const_gene_view
  gene_at(size_t index) const noexcept = 0;

  [[nodiscard]] virtual loss_type loss_of(const_gene_view) const noexcept = 0;

  // compute loss
  [[nodiscard]] loss_type loss_of(size_t index) const noexcept {
    return this->loss_of(this->gene_at(index));
  }

  [[nodiscard]] virtual size_t population_size() const noexcept = 0;

  virtual void loss_of_all(loss_matrix_type &result) const noexcept {
    result.setZero(this->num_objectives(), this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if constexpr (n_obj == 1) {
        result(idx) = this->loss_of(idx);
      } else {
        result.col(idx) = this->loss_of(idx);
      }
    }
  }

  [[nodiscard]] virtual loss_matrix_type loss_of_all() const noexcept {
    loss_matrix_type result;
    this->loss_of_all(result);
    return result;
  }
};

} // namespace cyka::genetic

#endif // CYKA_LOSS_COMPUTER_HPP
