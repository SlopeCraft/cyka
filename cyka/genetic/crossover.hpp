//
// Created by Joseph on 2024/5/3.
//

#ifndef CYKA_CROSSOVEROR_HPP
#define CYKA_CROSSOVEROR_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iterator>
#include <random>
#include <type_traits>

#include <Eigen/Dense>

#include "../utils/callback_fake_container.hpp"
#include "../utils/number_iterator.hpp"
#include "../utils/size_t_literal.hpp"

namespace cyka::genetic {

namespace detail {
struct empty_crossover_option {};
} // namespace detail

template <class mut_gene_view, class const_gene_view, class option_t>
class crossover_base {
public:
  static_assert(
      std::is_same_v<typename std::decay_t<mut_gene_view>::value_type,
                     typename std::decay_t<const_gene_view>::value_type>);
  virtual ~crossover_base() = default;

  virtual void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                         mut_gene_view d,
                         std::mt19937 &rand_engine) noexcept = 0;
  using crossover_option_type = option_t;
  using const_gene_view_type = const_gene_view;
  using mut_gene_view_type = mut_gene_view;

protected:
  crossover_option_type crossover_option_;

public:
  [[nodiscard]] const auto &crossover_option() const noexcept {
    return this->crossover_option_;
  }

  [[nodiscard]] virtual std::optional<std::invalid_argument>
  check_crossover_option(const crossover_option_type &opt) const noexcept {
    return std::nullopt;
  }

  void set_crossover_option(crossover_option_type &&opt) {
    auto result = this->check_crossover_option(opt);
    if (result) {
      throw(std::move(result.value()));
    }
    this->crossover_option_ = opt;
  }
};

template <class crossover_t>
concept is_crossover =
    requires(crossover_t &mut_c, const crossover_t &const_c,
             crossover_t::mut_gene_view_type mut_g,
             crossover_t::const_gene_view_type const_g,
             crossover_t::crossover_option_type opt, std::mt19937 &rand) {
      const_c.crossover_option();
      mut_c.set_crossover_option(std::move(opt));
      mut_c.crossover(const_g, const_g, mut_g, mut_g, rand);
    } and
    std::is_base_of_v<
        crossover_base<typename crossover_t::mut_gene_view_type,
                       typename crossover_t::const_gene_view_type,
                       typename crossover_t::crossover_option_type>,
        crossover_t>;

namespace detail {

// template <typename float_t, int64_t dim>
//   requires std::is_floating_point_v<float_t>
// void arithmetic_crossover(Eigen::Map<const Eigen::Array<float_t, dim, 1>> a,
//                           Eigen::Map<const Eigen::Array<float_t, dim, 1>> b,
//                           Eigen::Map<Eigen::Array<float_t, dim, 1>> c,
//                           Eigen::Map<Eigen::Array<float_t, dim, 1>> d,
//                           float_t ratio) noexcept {
template <class mut_gene_view, class const_gene_view>
void arithmetic_crossover(const const_gene_view &a, const const_gene_view &b,
                          mut_gene_view &c, mut_gene_view &d,
                          float_t ratio) noexcept {
  assert(a.rows() > 0);
  assert(a.rows() == b.rows());
  c = ratio * a + (1.0 - ratio) * b;
  d = ratio * b + (1.0 - ratio) * a;
  assert(c.size() == a.size());
  assert(c.size() == d.size());
}

template <class mut_gene_view, class const_gene_view>
void uniform_crossover(const const_gene_view &p1, const const_gene_view &p2,
                       mut_gene_view &c1, mut_gene_view &c2,
                       std::mt19937 &rand_engine, float probability) noexcept {
  assert(probability >= 0);
  assert(probability <= 1);
  assert(p1.size() == p2.size());
  c1.resize(p1.size());
  c2.resize(p1.size());

  std::uniform_real_distribution<float> rand{0, 1};
  for (ptrdiff_t i = 0; i < p1.size(); i++) {
    const float r = rand(rand_engine);
    const bool swap = r <= probability;
    if (swap) {
      c1[i] = p2[i];
      c2[i] = p1[i];
    } else {
      c1[i] = p1[i];
      c2[i] = p2[i];
    }
  }
}

template <class mut_gene_view, class const_gene_view>
void single_point_crossover(const const_gene_view &p1,
                            const const_gene_view &p2, mut_gene_view &c1,
                            mut_gene_view &c2, size_t point_index) noexcept {
  assert(p1.size() == p2.size());
  assert(point_index >= 0);
  assert(point_index < p1.size());
  c1.resize(p1.size());
  c2.resize(p1.size());

  for (ptrdiff_t i = 0; i < p1.size(); i++) {
    if (i < point_index) {
      c1[i] = p1[i];
      c2[i] = p2[i];
    } else {
      c1[i] = p2[i];
      c2[i] = p1[i];
    }
  }
}

template <class mut_gene_view, class const_gene_view>
void multi_point_crossover(const const_gene_view &p1, const const_gene_view &p2,
                           mut_gene_view &c1, mut_gene_view &c2,
                           size_t num_points, std::mt19937 &rand) noexcept {
  assert(p1.size() == p2.size());
  c1.resize(p1.size());
  c2.resize(p1.size());

  bool swap = false;
  ptrdiff_t prev{0};
  auto callback = [&p1, &p2, &c1, &c2, &swap, &prev](ptrdiff_t cur) {
    assert(cur < p1.size());
    assert(cur >= prev);
    const ptrdiff_t len = cur - prev;
    auto src1 = p1.segment(prev, len);
    auto src2 = p2.segment(prev, len);
    if (swap) {
      c1.segment(prev, len) = src2;
      c2.segment(prev, len) = src1;
    } else {
      c1.segment(prev, len) = src1;
      c2.segment(prev, len) = src2;
    }
    swap = not swap;
    prev = cur;
  };
  cyka::utils::callback_fake_container<decltype(callback), ptrdiff_t>
      fake_container{callback};
  std::sample(cyka::utils::number_iterator<ptrdiff_t>{0},
              cyka::utils::number_iterator<ptrdiff_t>{p1.size()},
              std::back_inserter(fake_container), num_points, rand);
}

} // namespace detail

template <typename float_type> struct arithmetic_crossover_option {
  float_type ratio{0.2};
};

template <class mut_gene_view, class const_gene_view>
class arithmetic_crossover
    : public crossover_base<
          mut_gene_view, const_gene_view,
                            arithmetic_crossover_option<typename std::decay_t<
                                mut_gene_view>::value_type>> {
public:
  using float_type = std::decay_t<mut_gene_view>::value_type;
  static_assert(std::is_floating_point_v<float_type>);

  [[nodiscard]] std::optional<std::invalid_argument>
  check_crossover_option(const arithmetic_crossover_option<float_type> &opt)
      const noexcept override {
    if (opt.ratio < 0 or opt.ratio > 1) {
      return std::invalid_argument{
          "ratio of arithmetic crossover should be in range [0,1]"};
    }
    return std::nullopt;
  }

  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &) noexcept override {
    detail::arithmetic_crossover(a, b, c, d, this->crossover_option().ratio);

    static_assert(is_crossover<std::decay_t<decltype(*this)>>);
  }
};

struct uniform_crossover_option {
  float swap_probability{0.5};
};
template <class mut_gene_view, class const_gene_view>
class uniform_crossover : public crossover_base<mut_gene_view, const_gene_view,
                                                uniform_crossover_option> {
public:
  [[nodiscard]] std::optional<std::invalid_argument> check_crossover_option(
      const uniform_crossover_option &opt) const noexcept override {
    if (opt.swap_probability < 0 or opt.swap_probability > 1) {
      return std::invalid_argument{"Swap probability should be in range [0,1]"};
    }
    return std::nullopt;
  }

  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    detail::uniform_crossover(a, b, c, d, mt,
                              this->crossover_option().swap_probability);
    static_assert(is_crossover<std::decay_t<decltype(*this)>>);
  }
};

template <class mut_gene_view, class const_gene_view>
class single_point_crossover
    : public crossover_base<mut_gene_view, const_gene_view,
                            detail::empty_crossover_option> {
public:
  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    assert(a.size() == b.size());
    std::uniform_int_distribution<ptrdiff_t> rand_idx(1, a.size() - 2);
    const ptrdiff_t idx = rand_idx(mt);
    detail::single_point_crossover(a, b, c, d,
                                   std::clamp<ptrdiff_t>(idx, 0, a.size() - 1));
    static_assert(is_crossover<std::decay_t<decltype(*this)>>);
  }
};

struct multi_crossover_option {
  size_t num_crossover_points{2};
};

template <class mut_gene_view, class const_gene_view>
class multi_point_crossover
    : public crossover_base<mut_gene_view, const_gene_view,
                            multi_crossover_option> {
public:
  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    assert(a.size() == b.size());

    detail::multi_point_crossover(
        a, b, c, d, this->crossover_option().num_crossover_points, mt);
    static_assert(is_crossover<std::decay_t<decltype(*this)>>);
  }
};

} // namespace cyka::genetic

#endif // CYKA_CROSSOVEROR_HPP
