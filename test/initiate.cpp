//
// Created by Joseph on 2024/5/1.
//
#include <Eigen/Dense>
#include <cyka/genetic/GA_system.hpp>
#include <cyka/genetic/crossover.hpp>
#include <cyka/genetic/mutator.hpp>
#include <cyka/genetic/population_in_map.hpp>
#include <cyka/genetic/population_in_matrix.hpp>
#include <cyka/genetic/population_in_vector.hpp>
#include <cyka/genetic/single_object_selector.hpp>
#include <cyka/genetic/solver_base.hpp>

#include <iostream>

void initiate_SO_selectors() noexcept;
void initiate_crossovers() noexcept;
void initiate_mutators() noexcept;

void initiate_GA_system() noexcept;

class square : public cyka::genetic::GA_system_base<
                   cyka::genetic::population_in_vector<Eigen::ArrayXf>, 1> {
public:
  [[nodiscard]] square::fitness_type
  fitness_of(square::const_gene_view_type g) const noexcept final {
    return g.matrix().squaredNorm();
  }
};

int main() {
  //  Eigen::Array<float, 5, 20> mat;
  //  //  mat.resize(5, 20);
  //  for (size_t i = 0; i < mat.size(); i++) {
  //    mat(i) = i;
  //  }
  //
  //  Eigen::Map<Eigen::Array<float, 5, 1>> m{&mat(0, 6), mat.rows()};
  //
  //  std::cout << "\nmat = \n" << mat;
  //  std::cout << "\nmat.col(6) = \n" << m;
  //
  //  return 0;
  cyka::genetic::population_in_matrix<float, 10> pop_10_1_col;
  cyka::genetic::population_in_map<Eigen::Array<float, 20, 1>> pop_map;
  cyka::genetic::population_in_vector<Eigen::Array<float, 20, 1>> pop_vec;

  cyka::genetic::GA_system<decltype(pop_10_1_col), 1> sys{
      [](auto) { return 0.0; }};
  initiate_SO_selectors();
  initiate_crossovers();
  initiate_mutators();
  initiate_GA_system();
}

void initiate_SO_selectors() noexcept {
  using namespace cyka::genetic::SO_selector;
  truncation trunc;
  tournament tour;
  monte_carlo mc;
  linear_rank lr;
  exponential_rank er;
  boltzmann b;
}

void initiate_crossovers() noexcept {
  using gene_t = Eigen::ArrayXd;
  cyka::genetic::arithmetic_crossover<Eigen::Map<gene_t>,
                                      Eigen::Map<const gene_t>>
      ac;
  cyka::genetic::uniform_crossover<Eigen::Map<gene_t>, Eigen::Map<const gene_t>>
      uc;
  cyka::genetic::single_point_crossover<Eigen::Map<gene_t>,
                                        Eigen::Map<const gene_t>>
      sc;
  cyka::genetic::multi_point_crossover<Eigen::Map<gene_t>,
                                       Eigen::Map<const gene_t>>
      mc;
}

void initiate_mutators() noexcept {
  using gene_t = Eigen::ArrayXd;
  cyka::genetic::arithmetic_mutator<Eigen::Map<gene_t>,
                                    Eigen::Map<const gene_t>>
      am;

  cyka::genetic::single_point_arithmetic_mutator<Eigen::Map<gene_t>,
                                                 Eigen::Map<const gene_t>>
      spam;

  cyka::genetic::single_point_boolean_mutator<
      Eigen::Map<Eigen::ArrayX<bool>>, Eigen::Map<const Eigen::ArrayX<bool>>>
      spbm;

  cyka::genetic::single_point_discrete_mutator<Eigen::Map<gene_t>,
                                               Eigen::Map<const gene_t>>
      spdm;
}

void initiate_GA_system() noexcept {
  square sys;

  using namespace cyka::genetic;
  using SOGA =
      single_object_GA<square, cyka::genetic::SO_selector::truncation,
                       arithmetic_crossover<square::mut_gene_view_type,
                                            square::const_gene_view_type>,
                       arithmetic_mutator<square::mut_gene_view_type,
                                          square::const_gene_view_type>>;

  SOGA solver{10};
  const size_t dims = 60;

  solver.set_GA_option(GA_option{.population_size = 400,
                                 .max_generations = 1000,
                                 .mutate_probability = 0.1});
  {
    SOGA::mutate_option_type opt;
    opt.lower_bound.setConstant(dims, -std::numeric_limits<float>::infinity());
    opt.upper_bound.setConstant(dims, std::numeric_limits<float>::infinity());
    opt.step_max.setConstant(dims, 0.5);
    solver.set_mutate_option(std::move(opt));
  }
  {
    std::normal_distribution<float> rand{10, 20};
    sys.reset(solver.GA_option().population_size,
              [&solver, &rand](Eigen::ArrayXf &f) {
                f.resize(dims);
                for (float &val : f) {
                  val = rand(solver.random_engine());
                }
              });
  }
  solver.set_crossover_option(SOGA::crossover_option_type{.ratio = 0.4});

  auto result = solver.optimize(sys);

  for (auto gen = 0zu; gen < result.fitness_history.size(); gen++) {
    auto &pair = result.fitness_history[gen];

    std::cout << "Generation " << gen << ", best fitness = "
              << pair.population_fitness[ptrdiff_t(pair.best_gene_index)]
              << "\n";
  }
}