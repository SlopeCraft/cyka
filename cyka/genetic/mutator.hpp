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
  using mut_gene_view_type = mut_gene_view;
  using const_gene_view_type = const_gene_view;

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
                      std::mt19937 &rand_engine) noexcept = 0;
};

template <class mutator_t>
concept is_mutator =
    requires(mutator_t &mut_m, const mutator_t &const_m,
             mutator_t::mut_gene_view_type mut_g,
             mutator_t::const_gene_view_type const_g,
             mutator_t::mutate_option_type opt, std::mt19937 &rand) {
      const_m.mutate_option();
      mut_m.set_mutate_option(std::move(opt));
      mut_m.mutate(const_g, mut_g, rand);
    } and
    std::is_base_of_v<mutator_base<typename mutator_t::mut_gene_view_type,
                                   typename mutator_t::const_gene_view_type,
                                   typename mutator_t::mutate_option_type>,
                      mutator_t>;

namespace detail {
template <class eigen_vec_t>
[[nodiscard]] std::optional<std::invalid_argument>
check_lb_ub(const eigen_vec_t &lb, const eigen_vec_t &ub) noexcept {

  if (lb.size() not_eq ub.size()) {
    return std::invalid_argument{
        "Lower and upper bound vector should have same size"};
  }

  if ((lb > ub).any()) {
    return std::invalid_argument{
        "Lower bound should be less or equal to upper bound"};
  }
  return std::nullopt;
}

template <class eigen_vec_t>
[[nodiscard]] std::optional<std::invalid_argument>
check_bound_and_step(const eigen_vec_t &lb, const eigen_vec_t &ub,
                     const eigen_vec_t &step_max) noexcept {
  if (auto err = check_lb_ub(lb, ub)) {
    return err;
  }

  if (lb.size() not_eq step_max.size()) {
    return std::invalid_argument{
        "step max, lower and upper bound vector should have same size"};
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

struct empty_mutate_option {};
} // namespace detail

template <typename float_t>
  requires std::is_floating_point_v<float_t>
struct arithmetic_mutate_option {
  Eigen::ArrayX<float_t> lower_bound{};
  Eigen::ArrayX<float_t> upper_bound{};
  Eigen::ArrayX<float_t> step_max{};
};

namespace detail {

template <class mut_gene_view, class const_gene_view>
void arithmetic_mutate(
    const_gene_view parent, mut_gene_view child, std::mt19937 &rand_engine,
    const arithmetic_mutate_option<typename mut_gene_view::value_type>
        &opt) noexcept {
  assert(parent.size() == opt.lower_bound.size());

  child.resize(parent.size());
  std::uniform_real_distribution<typename mut_gene_view::value_type> rand{-1.0,
                                                                          1.0};
  for (auto &f : child) {
    f = rand(rand_engine);
  }
  auto r_times_step = child * opt.step_max;
  auto new_val_before_clamp = r_times_step + parent;
  auto new_val = new_val_before_clamp.max(opt.lower_bound).min(opt.upper_bound);
  child = new_val;
}

template <class mut_gene_view, class const_gene_view>
void single_point_arithmetic_mutate(
    const_gene_view parent, mut_gene_view child, std::mt19937 &rand_engine,
    const arithmetic_mutate_option<typename mut_gene_view::value_type>
        &opt) noexcept {
  child = parent;
  std::uniform_int_distribution<ptrdiff_t> rand_idx{0, parent.size()};
  std::uniform_real_distribution<typename mut_gene_view::value_type> rand_f{-1,
                                                                            1};
  const ptrdiff_t idx = rand_idx(rand_engine);
  child[idx] += rand_f(rand_engine) * opt.step_max[idx];

  child = child.max(opt.lower_bound).min(opt.upper_bound);
  assert(child.size() == parent.size());
}
} // namespace detail

template <class mut_gene_view, class const_gene_view>
class arithmetic_mutator
    : public mutator_base<
          mut_gene_view, const_gene_view,
                          arithmetic_mutate_option<typename std::decay_t<
                              mut_gene_view>::value_type>> {
public:
  static_assert(
      std::is_same_v<typename std::decay_t<mut_gene_view>::value_type,
                     typename std::decay_t<const_gene_view>::value_type>);

  using float_type = std::decay_t<mut_gene_view>::value_type;
  std::optional<std::invalid_argument> check_mutate_option(
      const arithmetic_mutate_option<float_type> &opt) const noexcept override {
    return detail::check_bound_and_step(opt.lower_bound, opt.upper_bound,
                                        opt.step_max);
  }

  void mutate(const_gene_view parent, mut_gene_view child,
              std::mt19937 &rand_engine) noexcept override {
    detail::arithmetic_mutate(parent, child, rand_engine,
                              this->mutate_option());
    static_assert(is_mutator<std::decay_t<decltype(*this)>>);
  }
};

template <class mut_gene_view, class const_gene_view>
class single_point_arithmetic_mutator
    : public arithmetic_mutator<mut_gene_view, const_gene_view> {
public:
  void mutate(const_gene_view parent, mut_gene_view child,
              std::mt19937 &rand_engine) noexcept override {
    detail::single_point_arithmetic_mutate(parent, child, rand_engine,
                                           this->mutate_option());
  }
};

template <class mut_gene_view, class const_gene_view>
class single_point_boolean_mutator
    : public mutator_base<mut_gene_view, const_gene_view,
                          detail::empty_mutate_option> {
  static_assert(std::is_same_v<typename mut_gene_view::value_type, bool>);

public:
  void mutate(const_gene_view parent, mut_gene_view child,
              std::mt19937 &rand_engine) noexcept override {
    child = parent;
    std::uniform_int_distribution<ptrdiff_t> rand_idx{0, parent.size()};
    const auto index = rand_idx(rand_engine);
    child[index] = not child[index];
  }
};

template <typename scalar_type> struct discrete_mutate_option {
  Eigen::ArrayX<scalar_type> lower_bound;
  Eigen::ArrayX<scalar_type> upper_bound;
};

template <class mut_gene_view, class const_gene_view>
class single_point_discrete_mutator
    : public mutator_base<
          mut_gene_view, const_gene_view,
          discrete_mutate_option<typename mut_gene_view::value_type>> {
public:
  using scalar_type = mut_gene_view::value_type;
  [[nodiscard]] std::optional<std::invalid_argument> check_mutate_option(
      const discrete_mutate_option<scalar_type> &opt) const noexcept override {
    return detail::check_lb_ub(opt.lower_bound, opt.upper_bound);
  }

  void mutate(const_gene_view parent, mut_gene_view child,
              std::mt19937 &rand_engine) noexcept override {
    child = parent;

    std::uniform_real_distribution<float> rand_val{0, 1};
    auto get_rand_idx = [&rand_val, &rand_engine](int len) -> int {
      assert(len >= 0);
      const int ret = int(rand_val(rand_engine) * float(len));
      assert(ret < len or len == 0);
      return ret;
    };

    std::uniform_int_distribution<ptrdiff_t> rand_idx{0, parent.size()};
    const auto index = rand_idx(rand_engine);

    const scalar_type old_val = child[index];

    const scalar_type lb = this->mutate_option().lower_bound[index];
    const scalar_type ub = this->mutate_option().lower_bound[index];

    const scalar_type r = get_rand_idx(int(ub - lb) - 1);
    scalar_type new_val = scalar_type(r + lb);
    assert(new_val < ub);
    if (new_val >= r) {
      new_val += 1;
    }
    assert(lb <= new_val and new_val <= lb);

    child[index] = new_val;
  }
};

} // namespace cyka::genetic
#endif // CYKA_MUTATOR_HPP
