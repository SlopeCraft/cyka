//
// Created by Joseph on 2024/5/1.
//

#ifndef CYKA_POPULATION_IN_MATRIX_HPP
#define CYKA_POPULATION_IN_MATRIX_HPP

#include "population_base.hpp"

namespace cyka::genetic {

constexpr Eigen::Index map_n_objectives_to_rows(size_t n_obj) {
  if (n_obj == 0) {
    return Eigen::Dynamic;
  }
  return static_cast<Eigen::Index>(n_obj);
}

template <typename scalar_t, size_t n_features,
          int eigen_mat_option = Eigen::ColMajor>
class population_in_matrix
    : public population_base<
          Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>,
          Eigen::Map<
              Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>>,
          Eigen::Map<const Eigen::Array<
              scalar_t, map_n_objectives_to_rows(n_features), 1>>> {
public:
  using population_matrix_type =
      Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features),
                   Eigen::Dynamic, eigen_mat_option>;
  using base_type = population_base<
      Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>,
      Eigen::Map<
          Eigen::Array<scalar_t, map_n_objectives_to_rows(n_features), 1>>,
      Eigen::Map<const Eigen::Array<scalar_t,
                                    map_n_objectives_to_rows(n_features), 1>>>;
  using typename base_type::const_gene_view_type;
  using typename base_type::gene_type;
  using typename base_type::mut_gene_view_type;

  static_assert((eigen_mat_option & Eigen::RowMajorBit) == 0,
                "Storaging population in row major is not supported yet.");

protected:
  population_matrix_type gene_matrix;

public:
  [[nodiscard]] static const_gene_view_type
  wrap_col_const(const population_matrix_type &mat, size_t col) noexcept {
    return const_gene_view_type{&mat(0, col), mat.rows()};
  }
  [[nodiscard]] static const_gene_view_type
  wrap_col(const population_matrix_type &mat, size_t col) noexcept {
    return wrap_col_const(mat, col);
  }

  [[nodiscard]] static mut_gene_view_type wrap_col(population_matrix_type &mat,
                                                   size_t col) noexcept {
    return mut_gene_view_type{&mat(0, col), mat.rows()};
  }

  [[nodiscard]] size_t num_features() const noexcept {
    if constexpr (n_features > 0) {
      return n_features;
    }
    return this->gene_matrix.rows();
  }

  [[nodiscard]] size_t population_size() const noexcept override {
    return this->gene_matrix.cols();
  }
  [[nodiscard]] mut_gene_view_type gene_at(size_t idx) noexcept override {
    return wrap_col(this->gene_matrix, idx);
  }
  [[nodiscard]] const_gene_view_type
  gene_at(size_t idx) const noexcept override {
    return wrap_col(this->gene_matrix, idx);
  }

  void set_gene_at(size_t index, const_gene_view_type g) noexcept override {
    this->gene_matrix.col(index) = g;
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
      init_function(wrap_col(this->gene_matrix, 0));
    }

    for (size_t c = 1; c < num_population; c++) {
      init_function(wrap_col(this->gene_matrix, c));
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
    new_mat.setZero(this->num_features(), size_after);
    new_mat.block(0, 0, this->num_features(), size_before) = this->gene_matrix;
    //    for (size_t c = 0; c < size_before; c++) {
    //      new_mat.col(c) = this->gene_matrix.col(c);
    //    }

    size_t next_new_gene_idx = size_before;
    for (auto [aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx + 1 < size_after);
      crossover_function(wrap_col_const(new_mat, aidx),
                         wrap_col_const(new_mat, bidx),
                         wrap_col(new_mat, next_new_gene_idx),
                         wrap_col(new_mat, next_new_gene_idx + 1));
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
    new_mat.setZero(this->num_features(), size_after);
    new_mat.block(0, 0, this->num_features(), size_before) = this->gene_matrix;

    size_t next_new_gene_idx = size_before;
    for (size_t srcidx : mutate_list) {
      assert(srcidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx < size_after);
      mutate_function(wrap_col_const(new_mat, srcidx),
                      wrap_col(new_mat, next_new_gene_idx));
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
    new_mat.setZero(this->num_features(), size_after);
    new_mat.block(0, 0, this->num_features(), size_before) = this->gene_matrix;

    // crossover
    size_t next_new_gene_idx = size_before;
    for (auto [aidx, bidx] : crossover_list) {
      assert(aidx < size_before);
      assert(bidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx + 1 < size_after);
      crossover_function(wrap_col_const(new_mat, aidx),
                         wrap_col_const(new_mat, bidx),
                         wrap_col(new_mat, next_new_gene_idx),
                         wrap_col(new_mat, next_new_gene_idx + 1));
      next_new_gene_idx += 2;
    }
    // mutate
    for (size_t srcidx : mutate_list) {
      assert(srcidx < size_before);
      assert(next_new_gene_idx >= size_before);
      assert(next_new_gene_idx < size_after);
      mutate_function(wrap_col_const(new_mat, srcidx),
                      wrap_col(new_mat, next_new_gene_idx));
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
        new_gene_mat.col(c_write) = this->gene_matrix.col(c_read);
        c_write++;
      }
    }
    assert(c_write == left_pop_size);
    this->gene_matrix = left_pop_size;

    static_assert(is_population<population_in_matrix>);
  }
};
} // namespace cyka::genetic

#endif // CYKA_POPULATION_IN_MATRIX_HPP
