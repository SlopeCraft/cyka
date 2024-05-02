//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_SINGLE_OBJECT_SELECTOR_HPP
#define CYKA_SINGLE_OBJECT_SELECTOR_HPP

#include "selector.hpp"
#include <algorithm>
#include <map>

namespace cyka::genetic::SO_selector {

namespace detail {

void select_genes_by_score(Eigen::ArrayXd &probability_score,
                           Eigen::ArrayX<bool> &is_kept,
                           size_t num_to_eliminate,
                           std::mt19937 &rand_engine) noexcept {
  assert(num_to_eliminate < probability_score.size());
  const ptrdiff_t expected_group_size =
      probability_score.size() - ptrdiff_t(num_to_eliminate);
  assert(expected_group_size > 0);
  assert((probability_score >= 0).all());
  assert(probability_score.sum() > 0);
  is_kept.setConstant(probability_score.size(), true);

  std::uniform_real_distribution<double> rand{0, 1};
  size_t eliminated_num = 0;
  while (eliminated_num < num_to_eliminate) {
    double r = rand(rand_engine) * probability_score.sum();
    for (ptrdiff_t i = 0; i < probability_score.size(); i++) {
      if (!is_kept[i]) {
        assert(probability_score[i] == 0);
        //          continue;
      }
      r -= probability_score[i];

      if (r <= 0) { // selected
        probability_score[i] = 0;
        is_kept[i] = false;
        eliminated_num++;
        break;
      }
    }
    if (r > 0) [[unlikely]] {
      assert(("Impossible condition, r should be less or equal to the sum of "
              "whole array",
              0));
    }
  }

  assert(is_kept.count() == expected_group_size);
}

void sort_genes(std::span<const double> fitness,
                std::vector<size_t> &rank) noexcept {
  rank.resize(fitness.size());
  for (size_t i = 0; i < fitness.size(); i++) {
    rank[i] = i;
  }

  std::sort(rank.begin(), rank.end(), [&fitness](size_t i, size_t j) {
    return fitness[ptrdiff_t(i)] < fitness[ptrdiff_t(j)];
  });
}

[[nodiscard]] std::vector<size_t>
sort_genes(std::span<const double> fitness) noexcept {
  std::vector<size_t> dst;
  sort_genes(fitness, dst);
  return dst;
}

[[nodiscard]] std::vector<size_t>
sort_genes(const Eigen::Array<double, 1, Eigen::Dynamic> &fitness) noexcept {
  return sort_genes(
      std::span<const double>{fitness.data(), size_t(fitness.size())});
}

void select_ranked_genes(Eigen::ArrayXd &probability_score,
                         std::span<const size_t> rank,
                         Eigen::ArrayX<bool> &is_kept, size_t num_to_eliminate,
                         std::mt19937 &rand_engine) noexcept {
  assert(probability_score.size() == rank.size());
  assert(num_to_eliminate < probability_score.size());
  assert((probability_score >= 0).all());
  assert(probability_score.sum() > 0);
  const ptrdiff_t expected_group_size =
      probability_score.size() - ptrdiff_t(num_to_eliminate);
  assert(expected_group_size > 0);
  is_kept.setConstant(probability_score.size(), true);

  std::uniform_real_distribution<double> rand{0, 1};
  size_t num_eliminated = 0;
  while (num_eliminated < num_to_eliminate) {
    double r = rand(rand_engine) * probability_score.sum();
    for (ptrdiff_t sorted_idx = 0; sorted_idx < probability_score.size();
         sorted_idx++) {
      const auto gene_idx = (ptrdiff_t)rank[sorted_idx];
      if (!is_kept[gene_idx]) {
        assert(probability_score[sorted_idx] == 0);
      }

      r -= probability_score[sorted_idx];
      if (r <= 0) {
        probability_score[sorted_idx] = 0;
        is_kept[gene_idx] = false;
        num_eliminated++;
        break;
      }
    }

    if (r > 0) [[unlikely]] {
      assert(("Impossible condition, r should be less or equal to the sum of "
              "whole array",
              0));
    }
  }

  assert(is_kept.count() == expected_group_size);
}
} // namespace detail

class truncation : public selector_base<1> {
public:
  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<bool> &is_kept, std::mt19937 &) noexcept override {
    is_kept.resize(fitness.size());
    is_kept.fill(false);

    const auto rank = detail::sort_genes(fitness);
    for (size_t i = 0; i < expected_group_size; i++) {
      is_kept[ptrdiff_t(rank[i])] = true;
    }
  }
};

class linear_rank : public selector_base<1> {
public:
  struct option {
    double worst_probability{0.1};
    double best_probability{0.9};
  };

protected:
  option option_;

public:
  [[nodiscard]] auto &option() const noexcept { return this->option_; }

  void set_option(const struct option &opt) noexcept {
    const bool
        worst_gene_select_probability_should_be_less_than_the_best_gene_select_probability =
            opt.worst_probability < opt.best_probability;
    assert(
        worst_gene_select_probability_should_be_less_than_the_best_gene_select_probability);

    const bool probability_should_be_greater_or_equal_to_0 =
        opt.worst_probability >= 0 && opt.best_probability >= 0;
    assert(probability_should_be_greater_or_equal_to_0);

    const bool probability_should_be_less_or_equal_to_1 =
        opt.worst_probability <= 1 && opt.best_probability <= 1;
    assert(probability_should_be_less_or_equal_to_1);

    this->option_ = opt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<bool> &is_kept,
              std::mt19937 &rand_engine) noexcept override {

    const size_t num_to_eliminate = fitness.size() - expected_group_size;
    const size_t pop_size_before = fitness.cols();
    if (num_to_eliminate <= 0) {
      is_kept.resize(int64_t(pop_size_before));
      is_kept.fill(true);
      return;
    }
    assert(expected_group_size < fitness.size());
    assert(num_to_eliminate > 0);
    assert(pop_size_before > expected_group_size);

    const auto rank = detail::sort_genes(
        std::span<const double>{fitness.data(), size_t(fitness.size())});
    assert(rank.size() == fitness.size());

    Eigen::ArrayXd probability_score;
    assert(pop_size_before > 0);
    assert(this->option_.worst_probability >= 0);
    assert(this->option_.best_probability > this->option_.worst_probability);
    assert(this->option_.best_probability <= 1);
    probability_score.setLinSpaced(
        fitness.size(),
        this->option_.best_probability / double(pop_size_before),
        this->option_.worst_probability / double(pop_size_before));

    detail::select_ranked_genes(probability_score, rank, is_kept,
                                num_to_eliminate, rand_engine);
  }
};

class exponential_rank : public selector_base<1> {
public:
  struct option {
    double exponential_base = 0.8;
  };

protected:
  option option_;

public:
  [[nodiscard]] auto &option() const noexcept { return this->option_; }
  void set_option(struct option &opt) noexcept {

    const bool
        the_base_number_for_exponential_rank_selection_should_be_greater_or_equal_to_0 =
            opt.exponential_base >= 0;
    assert(
        the_base_number_for_exponential_rank_selection_should_be_greater_or_equal_to_0);

    const bool
        the_base_number_for_exponential_rank_selection_should_be_less_than_1 =
            opt.exponential_base < 1;
    assert(
        the_base_number_for_exponential_rank_selection_should_be_less_than_1);

    this->option_ = opt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<bool> &is_kept,
              std::mt19937 &rand_engine) noexcept override {
    assert(fitness.rows() == 1);
    assert(fitness.cols() >= expected_group_size);
    const size_t pop_size_before = fitness.cols();
    const size_t num_to_eliminate = pop_size_before - expected_group_size;
    is_kept.resize(int64_t(pop_size_before));
    is_kept.fill(true);
    if (num_to_eliminate <= 0) {
      return;
    }

    const std::vector<size_t> rank = detail::sort_genes(
        std::span<const double>{fitness.data(), size_t(fitness.size())});

    const double c_minus_1_div_c_pow_N_minus_1 =
        (this->option_.exponential_base - 1) /
        (std::pow(this->option_.exponential_base, pop_size_before) - 1);
    Eigen::ArrayXd probability;
    {
      probability.setConstant(ptrdiff_t(pop_size_before),
                              this->option_.exponential_base);
      Eigen::ArrayXd power;
      power.setLinSpaced(ptrdiff_t(pop_size_before), 0.0,
                         double(pop_size_before) - 1.0);
      probability = probability.pow(power) * c_minus_1_div_c_pow_N_minus_1;
    }

    detail::select_ranked_genes(probability, rank, is_kept, num_to_eliminate,
                                rand_engine);
  }
};

/**
 * \brief The Boltzmann method works by exponential function. The probability of
 * a gene is in proportion with exp(b*f), where b is the selection strength and
 * f is the fitness value.
 *
 * \note This method introduced one parameter: the selection strength b. For
 * maximum problems, b should be positive, while for minimize problems, b should
 * be a negative number.
 */
class boltzmann : public selector_base<1> {
public:
  struct option {
    double boltzmann_strength{-1.0};
  };

protected:
  option option_;

public:
  [[nodiscard]] auto &option() const noexcept { return this->option_; }
  void set_option(struct option &opt) noexcept {
    const bool
        boltzmann_strength_should_be_non_positive_because_less_fitness_is_considered_as_better =
            opt.boltzmann_strength <= 0;
    assert(
        boltzmann_strength_should_be_non_positive_because_less_fitness_is_considered_as_better);
    this->option_ = opt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<bool> &is_kept,
              std::mt19937 &rand_engine) noexcept override {
    assert(fitness.rows() == 1);
    assert(fitness.cols() >= expected_group_size);
    const size_t pop_size_before = fitness.cols();
    const size_t num_to_eliminate = pop_size_before - expected_group_size;
    is_kept.resize(int64_t(pop_size_before));
    is_kept.fill(true);
    if (num_to_eliminate <= 0) {
      return;
    }

    auto fitness_col = fitness.transpose();
    Eigen::ArrayXd probability_score =
        (fitness_col * this->option_.boltzmann_strength).exp();
    //    double score_sum = probability_score.sum();

    detail::select_genes_by_score(probability_score, is_kept, num_to_eliminate,
                                  rand_engine);
  }
};
} // namespace cyka::genetic::SO_selector

#endif // CYKA_SINGLE_OBJECT_SELECTOR_HPP
