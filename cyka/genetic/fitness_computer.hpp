//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_FITNESS_COMPUTER_HPP
#define CYKA_FITNESS_COMPUTER_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>

namespace cyka::genetic {

namespace detail {

// fixed multiple objectives
template <size_t n_objectives> class fitness_computer_impl {
public:
  using fitness_type = Eigen::Array<double, n_objectives, 1>;

  virtual ~fitness_computer_impl() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept {
    return n_objectives;
  }
};

template <> class fitness_computer_impl<1> {
public:
  using fitness_type = double;

  virtual ~fitness_computer_impl() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept { return 1; }
};

template <> class fitness_computer_impl<0> {
public:
  using fitness_type = Eigen::ArrayXd;

  virtual ~fitness_computer_impl() = default;

  [[nodiscard]] virtual size_t num_objectives() const noexcept = 0;
};
} // namespace detail

template <class const_gene_view, class fitness>
using fitness_function = fitness(const_gene_view);

template <size_t n_obj>
class fitness_computer : public detail::fitness_computer_impl<n_obj> {
public:
  using base_t = detail::fitness_computer_impl<n_obj>;
  using fitness_type = base_t::fitness_type;
  using fitness_matrix = Eigen::Array<double, n_obj, Eigen::Dynamic>;

  virtual ~fitness_computer() = default;

  // compute fitness
  [[nodiscard]] virtual fitness_type
  fitness_of(size_t index) const noexcept = 0;

  virtual void fitness_of_all(fitness_matrix &result) const noexcept {
    result.setZero(this->num_objectives(), this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if constexpr (n_obj == 1) {
        result(idx) = this->fitness_of(idx);
      } else {
        result.col(idx) = this->fitness_of(idx);
      }
    }
  }

  [[nodiscard]] virtual fitness_matrix fitness_of_all() const noexcept {
    fitness_matrix result;
    this->fitness_of_all(result);
    return result;
  }
};
} // namespace cyka::genetic

#endif // CYKA_FITNESS_COMPUTER_HPP
