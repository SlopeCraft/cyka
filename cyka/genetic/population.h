//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GABASE_H
#define HEURISTICFLOWR_GABASE_H

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>

#include <Eigen/Dense>

namespace cyka::genetic {

struct option {

  /**
   * \brief Size of the population. Default value is 100
   *
   */
  size_t population_size{100};

  /**
   * \brief GA will stop once best solution hasn't been improved for continuous
   * maxFailTimes generations. Default value is 100.
   *
   * \note This member doesn't works for MOGA solvers like NSGA2 and NSGA3 since
   * there's no proper way to estimate if the PF hasn't been changing for
   * generations.
   *
   */
  uint64_t early_stop_times{50};

  /**
   * \brief Maximum generation. GA will stop once reached this limitation.
   * Default value is 300
   *
   */
  uint64_t max_generations{300};

  /**
   * \brief Probability of a non-elite individual to join crossover. Default
   * value is 80%
   *
   */
  double crossover_probability{0.8};

  /**
   * \brief Probability of a non-elite individual to get mutated. Default value
   * is 5%
   */
  double mutate_probability{0.05};
};

// fixed multiple objectives
template <size_t n_objectives> class population_base {
public:
  using fitness_type = Eigen::Array<double, n_objectives, 1>;

  virtual ~population_base() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept {
    return n_objectives;
  }
};

template <> class population_base<1> {
public:
  using fitness_type = double;

  virtual ~population_base() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept { return 1; }
};

template <> class population_base<0> {
public:
  using fitness_type = Eigen::ArrayXd;

  virtual ~population_base() = default;

  [[nodiscard]] virtual size_t num_objectives() const noexcept = 0;
};

template <class const_gene_view, class fitness>
using fitness_function = fitness(const_gene_view);

template <class const_gene_view, class mut_gene_view>
using crossover_function = void(const_gene_view a, const_gene_view b,
                                mut_gene_view c, mut_gene_view d);

template <class const_gene_view, class mut_gene_view>
using mutate_function = void(const_gene_view src, mut_gene_view dest);

template <class gene, class mut_gene_view, class const_gene_view,
          size_t n_objectives>
  requires std::move_constructible<gene> && std::is_move_assignable_v<gene> &&
           std::is_default_constructible_v<gene> &&
           std::is_constructible_v<mut_gene_view, gene> &&
           std::is_constructible_v<const_gene_view, gene> &&
           std::is_constructible_v<const_gene_view, mut_gene_view> &&
           std::is_assignable_v<gene, mut_gene_view> &&
           std::is_assignable_v<gene, const_gene_view> &&
           std::is_assignable_v<mut_gene_view, const_gene_view>
class population : public population_base<n_objectives> {
public:
  using gene_type = gene;
  using mut_gene_view_type = mut_gene_view;
  using const_gene_view_type = const_gene_view;

  using base_t = population_base<n_objectives>;
  using fitness_type = base_t::fitness_type;
  using fitness_matrix = Eigen::Array<double, n_objectives, Eigen::Dynamic>;

  virtual ~population() = default;

  [[nodiscard]] virtual size_t population_size() const noexcept = 0;

  [[nodiscard]] virtual mut_gene_view gene_at(size_t index) noexcept = 0;
  [[nodiscard]] virtual const_gene_view
  gene_at(size_t index) const noexcept = 0;

  virtual void set_gene_at(size_t index, const_gene_view g) noexcept = 0;

  // reset
  virtual void
  reset(size_t num_population,
        const std::function<void(mut_gene_view)> &init_function) noexcept = 0;

  // compute fitness
  [[nodiscard]] virtual fitness_type
  fitness_of(size_t index) const noexcept = 0;

  virtual void fitness_of_all(fitness_matrix &result) const noexcept {
    result.setZero(this->num_objectives(), this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if constexpr (n_objectives == 1) {
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

  // crossover
  virtual void crossover(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function) noexcept = 0;

  // mutation
  virtual void
  mutate(std::span<const size_t> mutate_list,
         const std::function<mutate_function<const_gene_view, mut_gene_view>>
             &mutate_function) noexcept = 0;

  virtual void crossover_and_mutate(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function,
      std::span<const size_t> mutate_list,
      const std::function<mutate_function<const_gene_view, mut_gene_view>>
          &mutate_function) noexcept {
    this->crossover(crossover_list, crossover_function);
    this->mutate(mutate_list, mutate_function);
  }

  // selection
  virtual void select(std::span<const bool> LUT_is_selected) noexcept = 0;
};

template <class gene, class mut_gene_view, class const_gene_view,
          size_t n_objectives>
class population_in_map_impl
    : public population<gene, mut_gene_view, const_gene_view, n_objectives> {
protected:
  std::map<size_t, gene> gene_map;

public:
  [[nodiscard]] size_t population_size() const noexcept override {
    return this->gene_map.size();
  }
  [[nodiscard]] mut_gene_view gene_at(size_t idx) noexcept override {
    return this->gene_map.at(idx);
  }
  [[nodiscard]] const_gene_view gene_at(size_t idx) const noexcept override {
    return this->gene_map.at(idx);
  }

  void set_gene_at(size_t index, const_gene_view g) noexcept override {
    this->gene_map[index] = g;
  }

  void reset(size_t num_population, const std::function<void(mut_gene_view)>
                                        &init_function) noexcept override {
    this->gene_map.clear();
    for (size_t i; i < num_population; i++) {
      gene g{};
      init_function.value()(g);

      this->gene_map.emplace(i, std::move(g));
    }
  }

  void crossover(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function) noexcept override {
    const size_t size_before = this->gene_map.size();
    for (const auto &[aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      gene c{}, d{};
      crossover_function(this->gene_map.at(aidx), this->gene_map.at(bidx), c,
                         d);
      this->gene_map.emplace(this->gene_map.size(), std::move(c));
      this->gene_map.emplace(this->gene_map.size(), std::move(d));
    }
  }

  void
  mutate(std::span<const size_t> mutate_list,
         const std::function<mutate_function<const_gene_view, mut_gene_view>>
             &mutate_function) noexcept override {
    const size_t size_before = this->gene_map.size();
    for (size_t src_idx : mutate_list) {
      assert(src_idx < size_before);
      gene b{};
      mutate_function(this->gene_map.at(src_idx), b);
      this->gene_map.emplace(this->gene_map.size(), std::move(b));
    }
  }

  void select(std::span<const bool> LUT_is_selected) noexcept override {
    assert(LUT_is_selected.size() == this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if (!LUT_is_selected[idx]) {
        this->gene_map.erase(idx);
      }
    }

    size_t counter = 0;
    for (auto it = this->gene_map.begin();;) {
      if (it == this->gene_map.end()) {
        break;
      }
      assert(it->first >= counter);
      if (it->first == counter) {
        counter++;
        ++it;
        continue;
      }
      assert(it->first > counter);
      assert(!this->gene_map.contains(counter));

      this->gene_map.emplace(counter, std::move(it->second));
      it = this->gene_map.erase(it);
      counter++;
    }
  }
};

constexpr Eigen::Index map_n_objectives_to_rows(size_t n_obj) {
  if (n_obj == 0) {
    return Eigen::Dynamic;
  }
  return static_cast<Eigen::Index>(n_obj);
}

template <typename scalar_t, size_t n_features, size_t n_objectives,
          bool col_major = true>
class population_in_matrix
    : public population<
          Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>,
          Eigen::Map<
              Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>>,
          Eigen::Map<const Eigen::Array<
              scalar_t, map_n_objectives_to_rows(n_features), 1>>,
          n_features> {
public:
  using population_matrix_type =
      Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features),
                   Eigen::Dynamic,
                   col_major ? Eigen::ColMajor : Eigen::RowMajor>;
  using base_type = population<
      Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>,
      Eigen::Map<
          Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>>,
      Eigen::Map<const Eigen::Array<scalar_t,
                                    map_n_objectives_to_rows(n_features), 1>>,
      n_objectives>;
  using typename base_type::const_gene_view_type;
  using typename base_type::gene_type;
  using typename base_type::mut_gene_view_type;

protected:
  population_matrix_type gene_matrix;

public:
  [[nodiscard]] size_t population_size() const noexcept override {
    return this->gene_matrix.cols();
  }
  [[nodiscard]] mut_gene_view_type gene_at(size_t idx) noexcept override {
    return this->gene_map.col(idx);
  }
  [[nodiscard]] const_gene_view_type
  gene_at(size_t idx) const noexcept override {
    return this->gene_map.col(idx);
  }

  void set_gene_at(size_t index, const_gene_view_type g) noexcept override {
    this->gene_map.col(index) = g;
  }

  void reset(size_t num_population,
             const std::function<void(mut_gene_view_type)>
                 &init_function) noexcept override {
    assert(num_population > 0);
    if constexpr (n_features <= 0) { // dynamic features
      gene_type g{};
      init_function(g);
      const size_t num_features = const_gene_view_type{g}.rows();
      this->gene_matrix.setZero(num_features, num_population);
      gene_matrix.col(0) = std::move(g);
    } else { // fixed features
      this->gene_matrix.setZero(n_features, num_population);
      init_function(this->gene_matrix.col(0));
    }

    for (size_t c = 1; c < num_population; c++) {
      init_function(this->gene_matrix.col(c));
    }
  }

  void
  crossover(std::span<const std::pair<size_t, size_t>> crossover_list,
            const std::function<
                crossover_function<const_gene_view_type, mut_gene_view_type>>
                &crossover_function) noexcept override {
    const size_t size_before = this->population_size();
    const size_t size_after = size_before + 2 * crossover_list.size();
    population_matrix_type new_mat;
    new_mat.setZero(this->num_objectives(), size_after);
    new_mat.block(0, 0, this->num_objectives(), size_before) =
        this->gene_matrix;
    //    for (size_t c = 0; c < size_before; c++) {
    //      new_mat.col(c) = this->gene_matrix.col(c);
    //    }

    size_t next_new_gene_idx = size_before;
    for (auto [aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx + 1 < size_after);
      crossover_function(new_mat.col(aidx), new_mat.col(bidx),
                         new_mat.col(next_new_gene_idx),
                         new_mat.col(next_new_gene_idx + 1));
      next_new_gene_idx += 2;
    }
    this->gene_matrix = new_mat;
  }

  void mutate(std::span<const size_t> mutate_list,
              const std::function<
                  mutate_function<const_gene_view_type, mut_gene_view_type>>
                  &mutate_function) noexcept override {
    const size_t size_before = this->population_size();
    const size_t size_after = size_before + mutate_list.size();
    population_matrix_type new_mat;
    new_mat.setZero(this->num_objectives(), size_after);
    new_mat.block(0, 0, this->num_objectives(), size_before) =
        this->gene_matrix;

    size_t next_new_gene_idx = size_before;
    for (size_t srcidx : mutate_list) {
      assert(srcidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx < size_after);
      mutate_function(new_mat.col(srcidx), new_mat.col(next_new_gene_idx));
      next_new_gene_idx++;
    }
    this->gene_matrix = new_mat;
  }

  void crossover_and_mutate(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<
          crossover_function<const_gene_view_type, mut_gene_view_type>>
          &crossover_function,
      std::span<const size_t> mutate_list,
      const std::function<
          mutate_function<const_gene_view_type, mut_gene_view_type>>
          &mutate_function) noexcept override {

    const size_t size_before = this->population_size();
    const size_t size_after =
        size_before + 2 * crossover_list.size() + mutate_list.size();

    population_matrix_type new_mat;
    new_mat.setZero(this->num_objectives(), size_after);
    new_mat.block(0, 0, this->num_objectives(), size_before) =
        this->gene_matrix;

    // crossover
    size_t next_new_gene_idx = size_before;
    for (auto [aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx + 1 < size_after);
      crossover_function(new_mat.col(aidx), new_mat.col(bidx),
                         new_mat.col(next_new_gene_idx),
                         new_mat.col(next_new_gene_idx + 1));
      next_new_gene_idx += 2;
    }
    // mutate
    for (size_t srcidx : mutate_list) {
      assert(srcidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx < size_after);
      mutate_function(new_mat.col(srcidx), new_mat.col(next_new_gene_idx));
      next_new_gene_idx++;
    }

    this->gene_matrix = new_mat;
  }

  void select(std::span<const bool> LUT_is_selected) noexcept override {
    assert(LUT_is_selected.size() == this->population_size());
    const size_t left_pop_size = [&LUT_is_selected]() {
      size_t result = 0;
      for (bool select : LUT_is_selected) {
        if (select) {
          result++;
        }
      }
      return result;
    }();

    population_matrix_type new_gene_mat;
    new_gene_mat.setZero(this->gene_matrix.rows(), left_pop_size);

    size_t c_write = 0;
    for (size_t c_read = 0; c_read < this->gene_matrix.cols(); c_read++) {
      if (LUT_is_selected[c_read]) {
        new_gene_mat[c_write] = this->gene_matrix[c_read];
        c_write++;
      }
    }
    assert(c_write == left_pop_size);
    this->gene_matrix = left_pop_size;
  }
};

} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GABASE_H
