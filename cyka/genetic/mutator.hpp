//
// Created by Joseph on 2024/5/5.
//

#ifndef CYKA_MUTATOR_HPP
#define CYKA_MUTATOR_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <random>
#include <type_traits>

#include <Eigen/Dense>

namespace cyka::genetic {

template <class mut_gene_view, class const_gene_view, class option_t>
class mutator_base {
public:
  using mutate_option_type = option_t;

protected:
  mutate_option_type mutate_option_;

public:
  [[nodiscard]] const auto &mutate_option() const noexcept {
    return this->mutate_option_;
  }

  [[nodiscard]] virtual std::optional<std::invalid_argument>
  check_mutate_option(const mutate_option_type &) const noexcept {
    return std::nullopt;
  }

  void set_mutate_option(mutate_option_type &&opt) {
    auto err = this->check_mutate_option(opt);
    if (err) {
      throw std::move(err.value());
    }
    this->mutate_option_ = opt;
  }

  virtual void mutate(const_gene_view parent, mut_gene_view child,
                      std::mt19937 &rand_engine) const noexcept = 0;
};

namespace detail {
template <class eigen_vec_t>
[[nodiscard]] std::optional<std::invalid_argument>
check_bound_and_step(const eigen_vec_t &lb, const eigen_vec_t &ub,
                     const eigen_vec_t &step_max) noexcept {
  if ((lb.size() not_eq ub.size()) or (lb.size() not_eq step_max.size())) {
    return std::invalid_argument{
        "step max, lower and upper bound vector should have same size"};
  }

  if ((lb > ub).any()) {
    return std::invalid_argument{
        "Lower bound should be less or equal to upper bound"};
  }
  if ((step_max < 0).any()) {
    return std::invalid_argument{"step_max should be in range [0,+inf)"};
  }
  if (lb.hasNaN() or ub.hasNaN() or step_max.hasNaN()) {
    return std::invalid_argument{
        "step max, lower and upper bound vector shouldn't contain nan"};
  }
  if (not(step_max.isFinite()).all()) {
    return std::invalid_argument{"step max should be finite"};
  }

  return std::nullopt;
}
} // namespace detail

template <typename float_t>
  requires std::is_floating_point_v<float_t>
struct arithmetic_mutate_option {
  Eigen::ArrayX<float_t> lower_bound{};
  Eigen::ArrayX<float_t> upper_bound{};
  Eigen::ArrayX<float_t> step_max{};
};

template <class mut_gene_view, class const_gene_view>
class arithmetic_mutator
    : public mutator_base<
          mut_gene_view, const_gene_view,
          arithmetic_mutate_option<typename mut_gene_view::value_type>> {
public:
  static_assert(std::is_same_v<typename mut_gene_view::value_type,
                               typename const_gene_view::value_type>);

  using float_type = mut_gene_view::value_type;
  std::optional<std::invalid_argument> check_mutate_option(
      const arithmetic_mutate_option<float_type> &opt) const noexcept override {
    return detail::check_bound_and_step(opt.lower_bound, opt.upper_bound,
                                        opt.step_max);
  }

  void mutate(const_gene_view parent, mut_gene_view child,
              std::mt19937 &rand_engine) const noexcept override {
    assert(parent.size() == this->mutate_option().lower_bound.size());

    child.resize(parent.size());
    std::uniform_real_distribution<float_type> rand{-1.0, 1.0};
    for (float_type &f : child) {
      f = rand(rand_engine);
    }
    auto r_times_step = child * this->mutate_option().step_max;
    auto new_val_before_clamp = r_times_step + parent;
    auto new_val = new_val_before_clamp.max(this->mutate_option().lower_bound)
                       .min(this->mutate_option().upper_bound);
    child = new_val;
  }
};

} // namespace cyka::genetic
#endif // CYKA_MUTATOR_HPP
