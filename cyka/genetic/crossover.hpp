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

#include <Eigen/Dense>

namespace cyka::genetic {

template <class mut_gene_view, class const_gene_view> class crossover_base {
public:
  static_assert(std::is_same_v<typename mut_gene_view::value_type,
                               typename const_gene_view::value_type>);
  virtual ~crossover_base() = default;

  virtual void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                         mut_gene_view d,
                         std::mt19937 &rand_engine) noexcept = 0;
};

namespace detail {

// template <typename float_t, int64_t dim>
//   requires std::is_floating_point_v<float_t>
// void arithmetic_crossover(Eigen::Map<const Eigen::Array<float_t, dim, 1>> a,
//                           Eigen::Map<const Eigen::Array<float_t, dim, 1>> b,
//                           Eigen::Map<Eigen::Array<float_t, dim, 1>> c,
//                           Eigen::Map<Eigen::Array<float_t, dim, 1>> d,
//                           float_t ratio) noexcept {
template <class mut_gene_view, class const_gene_view>
void arithmetic_crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                          mut_gene_view d, float_t ratio) noexcept {
  assert(a.rows() == b.rows());
  c = ratio * a + (1.0 - ratio) * b;
  d = ratio * b + (1.0 - ratio) * a;
  assert(c.size() == d.size());
}

template <class mut_gene_view, class const_gene_view>
void uniform_crossover(const_gene_view p1, const_gene_view p2, mut_gene_view c1,
                       mut_gene_view c2, std::mt19937 &rand_engine,
                       float probability) noexcept {
  assert(probability >= 0);
  assert(probability <= 1);
  assert(p1.size() == p2.size());
  c1.resize(p1.size());
  c2.resize(p1.size());

  std::uniform_real_distribution<float> rand(0, 1);
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
void single_point_crossover(const_gene_view p1, const_gene_view p2,
                            mut_gene_view c1, mut_gene_view c2,
                            size_t point_index) noexcept {
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
void multi_point_crossover(const_gene_view p1, const_gene_view p2,
                           mut_gene_view c1, mut_gene_view c2,
                           size_t num_points, std::mt19937 &rand) noexcept {
  assert(p1.size() == p2.size());
  c1.resize(p1.size());
  c2.resize(p1.size());

  // iterator to receive
  struct sampler_it {
    using iterator_category = std::output_iterator_tag;

    using value_type = ptrdiff_t;
    using difference_type = ptrdiff_t;
    using pointer = ptrdiff_t *;
    using reference = ptrdiff_t &;
    ptrdiff_t prev{0};
    ptrdiff_t cur{-1};
    bool swap{true};
    const_gene_view *p1;
    const_gene_view *p2;
    mut_gene_view *c1;
    mut_gene_view *c2;
    sampler_it(const_gene_view *p1, const_gene_view *p2, mut_gene_view *c1,
               mut_gene_view *c2)
        : p1{p1}, p2{p2}, c1{c1}, c2{c2} {};

    ptrdiff_t &operator*() noexcept { return this->cur; }

    sampler_it &operator++() noexcept {
      assert(cur < this->p1->size());
      assert(cur >= prev);
      const ptrdiff_t len = cur - prev;
      auto src1 = p1->segment(prev, len);
      auto src2 = p2->segment(prev, len);
      if (swap) {
        c1->segment(prev, len) = src2;
        c2->segment(prev, len) = src1;
      } else {
        c1->segment(prev, len) = src1;
        c2->segment(prev, len) = src2;
      }
      this->swap = not this->swap;
      this->prev = this->cur;
      this->cur = -1;

      return *this;
    }
  };

  struct index_it {
    using iterator_category = std::forward_iterator_tag;
    using value_type = ptrdiff_t;
    using difference_type = ptrdiff_t;
    using pointer = ptrdiff_t *;
    using reference = ptrdiff_t &;
    ptrdiff_t idx;
    ptrdiff_t &operator*() noexcept { return this->idx; }
    index_it &operator++() noexcept {
      this->idx++;
      return *this;
    }
    bool operator!=(index_it b) const noexcept {
      return this->idx not_eq b.idx;
    }
  };

  std::sample(index_it{.idx = 0}, index_it{.idx = p1.size()},
              sampler_it{&p1, &p2, &c1, &c2}, num_points, rand);
}

} // namespace detail

template <class mut_gene_view, class const_gene_view>
class arithmetic_crossover
    : public crossover_base<mut_gene_view, const_gene_view> {
public:
  using float_type = mut_gene_view::value_type;
  static_assert(std::is_floating_point_v<float_type>);
  struct crossover_option {
    float_type ratio{0.2};
  };

protected:
  crossover_option crossover_option_;

public:
  [[nodiscard]] const auto &crossover_option() const noexcept {
    return this->crossover_option_;
  }
  void set_crossover_option(const struct crossover_option &opt) noexcept {
    assert(opt.ratio >= 0);
    assert(opt.ratio <= 1);
    this->crossover_option_ = opt;
  }

  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &) noexcept override {
    detail::arithmetic_crossover(a, b, c, d, this->crossover_option_.ratio);
  }
};

template <class mut_gene_view, class const_gene_view>
class uniform_crossover
    : public crossover_base<mut_gene_view, const_gene_view> {
public:
  struct crossover_option {
    float swap_probability{0.5};
  };

protected:
  crossover_option crossover_option_;

public:
  [[nodiscard]] const auto &crossover_option() const noexcept {
    return this->crossover_option_;
  }
  void set_crossover_option(const struct crossover_option &opt) noexcept {
    assert(opt.swap_probability >= 0);
    assert(opt.swap_probability <= 1);
    this->crossover_option_ = opt;
  }

  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    detail::uniform_crossover(a, b, c, d, mt,
                              this->crossover_option().swap_probability);
  }
};

template <class mut_gene_view, class const_gene_view>
class single_point_crossover
    : public crossover_base<mut_gene_view, const_gene_view> {
public:
  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    assert(a.size() == b.size());
    std::uniform_int_distribution<ptrdiff_t> rand_idx(1, a.size() - 2);
    const ptrdiff_t idx = rand_idx(mt);
    detail::single_point_crossover(a, b, c, d,
                                   std::clamp<ptrdiff_t>(idx, 0, a.size() - 1));
  }
};
template <class mut_gene_view, class const_gene_view>
class multi_point_crossover
    : public crossover_base<mut_gene_view, const_gene_view> {
public:
  struct crossover_option {
    size_t num_crossover_points{2};
  };

protected:
  crossover_option crossover_option_;

public:
  [[nodiscard]] const auto &crossover_option() const noexcept {
    return this->crossover_option_;
  }
  void set_crossover_option(const struct crossover_option &opt) noexcept {
    this->crossover_option_ = opt;
  }
  void crossover(const_gene_view a, const_gene_view b, mut_gene_view c,
                 mut_gene_view d, std::mt19937 &mt) noexcept override {
    assert(a.size() == b.size());

    detail::multi_point_crossover(
        a, b, c, d, this->crossover_option_.num_crossover_points, mt);
  }
};

} // namespace cyka::genetic

#endif // CYKA_CROSSOVEROR_HPP
