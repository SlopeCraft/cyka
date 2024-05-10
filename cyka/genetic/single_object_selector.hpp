//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_SINGLE_OBJECT_SELECTOR_HPP
#define CYKA_SINGLE_OBJECT_SELECTOR_HPP

#include <algorithm>
#include <map>

#include "../utils/number_iterator.hpp"
#include "selector.hpp"

namespace cyka::genetic::SO_selector {

namespace detail {

inline void select_genes_by_score(Eigen::ArrayXd &probability_score,
                                  Eigen::ArrayX<uint16_t> &selected_count,
                                  size_t num_to_eliminate,
                           std::mt19937 &rand_engine) noexcept {
  assert(num_to_eliminate < probability_score.size());
  const ptrdiff_t expected_group_size =
      probability_score.size() - ptrdiff_t(num_to_eliminate);
  assert(expected_group_size > 0);
  assert((probability_score >= 0).all());
  assert(probability_score.sum() > 0);
  selected_count.setConstant(probability_score.size(), 1);

  std::uniform_real_distribution<double> rand{0, 1};
  size_t eliminated_num = 0;
  while (eliminated_num < num_to_eliminate) {
    double r = rand(rand_engine) * probability_score.sum();
    for (ptrdiff_t i = 0; i < probability_score.size(); i++) {
      if (not selected_count[i]) {
        assert(probability_score[i] == 0);
        //          continue;
      }
      r -= probability_score[i];

      if (r <= 0) { // selected
        probability_score[i] = 0;
        selected_count[i] = 0;
        eliminated_num++;
        break;
      }
    }
    if (r > 0) [[unlikely]] {
      const bool
          impossible_condition_r_should_be_less_or_equal_to_the_sum_of_the_whole_array =
              false;
      assert(
          impossible_condition_r_should_be_less_or_equal_to_the_sum_of_the_whole_array);
    }
  }

  assert(selected_count.sum() == expected_group_size);
}

inline void sort_genes(std::span<const double> fitness,
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
                         Eigen::ArrayX<uint16_t> &selected_count,
                         size_t num_to_eliminate,
                         std::mt19937 &rand_engine) noexcept {
  assert(probability_score.size() == rank.size());
  assert(num_to_eliminate < probability_score.size());
  assert((probability_score >= 0).all());
  assert(probability_score.sum() > 0);
  const ptrdiff_t expected_group_size =
      probability_score.size() - ptrdiff_t(num_to_eliminate);
  assert(expected_group_size > 0);
  selected_count.setConstant(probability_score.size(), 1);

  std::uniform_real_distribution<double> rand{0, 1};
  size_t num_eliminated = 0;
  while (num_eliminated < num_to_eliminate) {
    double r = rand(rand_engine) * probability_score.sum();
    for (ptrdiff_t sorted_idx = 0; sorted_idx < probability_score.size();
         sorted_idx++) {
      const auto gene_idx = (ptrdiff_t)rank[sorted_idx];
      if (not selected_count[gene_idx]) {
        assert(probability_score[sorted_idx] == 0);
      }

      r -= probability_score[sorted_idx];
      if (r <= 0) {
        probability_score[sorted_idx] = 0;
        selected_count[gene_idx] = false;
        num_eliminated++;
        break;
      }
    }

    if (r > 0) [[unlikely]] {
      const bool
          impossible_condition_r_should_be_less_or_equal_to_the_sum_of_the_whole_array =
              false;
      assert(
          impossible_condition_r_should_be_less_or_equal_to_the_sum_of_the_whole_array);
    }
  }

  assert(selected_count.sum() == expected_group_size);
}

struct empty_option {};

} // namespace detail

class roulette_wheel : public selector_base<1, detail::empty_option> {
public:
  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand) noexcept override {
    assert(expected_group_size <= fitness.size());
    Eigen::ArrayXd scores = fitness.maxCoeff() - fitness.transpose();

    detail::select_genes_by_score(scores, selected_count,
                                  fitness.size() - expected_group_size, rand);

    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }

  [[nodiscard]] std::optional<std::invalid_argument>
  check_select_option(const detail::empty_option &opt) const noexcept override {
    return std::nullopt;
  }
};

struct tournament_option {
  size_t tournament_size = 3;
};

class tournament : public selector_base<1, tournament_option> {
public:
  [[nodiscard]] std::optional<std::invalid_argument>
  check_select_option(const tournament_option &opt) const noexcept override {
    if (opt.tournament_size < 1) {
      return std::invalid_argument{"tournament size should be positive number"};
    }
    return std::nullopt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand_engine) noexcept override {
    const size_t pop_size_before = fitness.size();
    selected_count.setZero(fitness.size());

    auto choose_tournament = [this, pop_size_before,
                              &rand_engine](std::vector<size_t> &src_indices) {
      src_indices.clear();
      src_indices.reserve(this->select_option().tournament_size);
      std::sample(cyka::utils::number_iterator<size_t>(0),
                  cyka::utils::number_iterator<size_t>(pop_size_before),
                  std::back_inserter(src_indices),
                  this->select_option().tournament_size, rand_engine);
    };
    std::vector<size_t> tournament;
    Eigen::ArrayXd tournament_fitness;
    for (auto num_selected = 0zu; num_selected < expected_group_size;
         num_selected++) {
      choose_tournament(tournament);
      //      assert(tournament.size() ==
      //      this->select_option().tournament_size);
      tournament_fitness.setZero((ptrdiff_t)tournament.size());
      for (ptrdiff_t tournament_idx = 0; tournament_idx < tournament.size();
           tournament_idx++) {
        tournament_fitness[tournament_idx] =
            fitness[(ptrdiff_t)tournament[tournament_idx]];
      }

      Eigen::Index best_in_tournament = -1;
      tournament_fitness.minCoeff(&best_in_tournament);
      assert(best_in_tournament >= 0);
      assert(best_in_tournament < fitness.size());
      selected_count[best_in_tournament]++;
    }

    assert(selected_count.sum() == expected_group_size);
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};

class monte_carlo : public selector_base<1, detail::empty_option> {
public:
  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand) noexcept override {
    assert(expected_group_size <= fitness.size());
    Eigen::ArrayXd scores;
    scores.setOnes(fitness.size());
    detail::select_genes_by_score(scores, selected_count,
                                  fitness.size() - expected_group_size, rand);
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};

class truncation : public selector_base<1, detail::empty_option> {
public:
  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &) noexcept override {
    selected_count.resize(fitness.size());
    selected_count.fill(0);

    const auto rank = detail::sort_genes(fitness);
    for (size_t i = 0; i < expected_group_size; i++) {
      selected_count[ptrdiff_t(rank[i])] = 1;
    }
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};

struct linear_rank_option {
  double worst_probability{0.1};
  double best_probability{0.9};
};

class linear_rank : public selector_base<1, linear_rank_option> {
public:
  [[nodiscard]] std::optional<std::invalid_argument>
  check_select_option(const select_option_type &opt) const noexcept override {
    if (opt.worst_probability >= opt.best_probability) {
      return std::invalid_argument{
          "worst gene select probability should be less than the best gene "
          "select probability"};
    }
    if (opt.worst_probability < 0) {
      return std::invalid_argument{
          "Probability should be greater or equal to 0"};
    }
    if (opt.best_probability > 1) {
      return std::invalid_argument{"Probability should be less or equal to 1"};
    }
    return std::nullopt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand_engine) noexcept override {

    const size_t num_to_eliminate = fitness.size() - expected_group_size;
    const size_t pop_size_before = fitness.cols();
    if (num_to_eliminate <= 0) {
      selected_count.resize(int64_t(pop_size_before));
      selected_count.fill(1);
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
    assert(this->select_option().worst_probability >= 0);
    assert(this->select_option().best_probability >
           this->select_option().worst_probability);
    assert(this->select_option().best_probability <= 1);
    probability_score.setLinSpaced(
        fitness.size(),
        this->select_option().best_probability / double(pop_size_before),
        this->select_option().worst_probability / double(pop_size_before));

    detail::select_ranked_genes(probability_score, rank, selected_count,
                                num_to_eliminate, rand_engine);
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};

struct exponential_rank_option {
  double exponential_base = 0.8;
};

class exponential_rank : public selector_base<1, exponential_rank_option> {
public:
  [[nodiscard]] std::optional<std::invalid_argument>
  check_select_option(const select_option_type &opt) const noexcept override {
    if (opt.exponential_base < 0 || opt.exponential_base >= 1) {
      return std::invalid_argument{"The base number for exponential rank "
                                   "selection should be in range [0,1)"};
    }
    return std::nullopt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand_engine) noexcept override {
    assert(fitness.rows() == 1);
    assert(fitness.cols() >= expected_group_size);
    const size_t pop_size_before = fitness.cols();
    const size_t num_to_eliminate = pop_size_before - expected_group_size;
    selected_count.resize(int64_t(pop_size_before));
    selected_count.fill(1);
    if (num_to_eliminate <= 0) {
      return;
    }

    const std::vector<size_t> rank = detail::sort_genes(
        std::span<const double>{fitness.data(), size_t(fitness.size())});

    const double c_minus_1_div_c_pow_N_minus_1 =
        (this->select_option().exponential_base - 1) /
        (std::pow(this->select_option().exponential_base, pop_size_before) - 1);
    Eigen::ArrayXd probability;
    {
      probability.setConstant(ptrdiff_t(pop_size_before),
                              this->select_option().exponential_base);
      Eigen::ArrayXd power;
      power.setLinSpaced(ptrdiff_t(pop_size_before), 0.0,
                         double(pop_size_before) - 1.0);
      probability = probability.pow(power) * c_minus_1_div_c_pow_N_minus_1;
    }

    detail::select_ranked_genes(probability, rank, selected_count,
                                num_to_eliminate, rand_engine);
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};

struct boltzmann_option {
  double boltzmann_strength{-1.0};
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
class boltzmann : public selector_base<1, boltzmann_option> {
public:
  [[nodiscard]] std::optional<std::invalid_argument>
  check_select_option(const select_option_type &opt) const noexcept override {
    if (opt.boltzmann_strength > 0) {
      return std::invalid_argument{"Boltzmann strength should be <=0"};
    }
    return std::nullopt;
  }

  void select(const fitness_matrix &fitness, size_t expected_group_size,
              Eigen::ArrayX<uint16_t> &selected_count,
              std::mt19937 &rand_engine) noexcept override {
    assert(fitness.rows() == 1);
    assert(fitness.cols() >= expected_group_size);
    const size_t pop_size_before = fitness.cols();
    const size_t num_to_eliminate = pop_size_before - expected_group_size;
    selected_count.resize(int64_t(pop_size_before));
    selected_count.fill(1);
    if (num_to_eliminate <= 0) {
      return;
    }

    auto fitness_col = fitness.transpose();
    Eigen::ArrayXd probability_score =
        (fitness_col * this->select_option().boltzmann_strength).exp();
    //    double score_sum = probability_score.sum();

    detail::select_genes_by_score(probability_score, selected_count,
                                  num_to_eliminate, rand_engine);
    static_assert(is_selector<std::decay_t<decltype(*this)>>);
  }
};
} // namespace cyka::genetic::SO_selector

#endif // CYKA_SINGLE_OBJECT_SELECTOR_HPP
