/* ---------------------------------------------------------------------
*
* Copyright (C) 2000 - 2015 by the deal.II authors
*
* This file is part of the deal.II library.
*
* The deal.II library is free software; you can use it, redistribute
* it, and/or modify it under the terms of the GNU Lesser General
* Public License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
* The full text of the license can be found in the file LICENSE at
* the top level of the deal.II distribution.
*
* ---------------------------------------------------------------------

*
* Author: Wolfgang Bangerth, University of Heidelberg, 2000
*/


const bool PRINT_ALGEBRA = false;
// The following means literaly "1 Amper"
const double TOTAL_CURRENT = 1.;
// Init ref
const unsigned int INIT_REF_NUM = 6;
const double M_PI = 3.141592654;
#define INCLUDE_TOTAL_CURRENT 0

// @sect3{Include files}

// As usual, the first few include files are already known, so we will not
// comment on them further.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <fstream>
#include <iostream>

using namespace dealii;

bool isOutsideOfSmallSquare(dealii::Point<2> point)
{
  return (std::abs(point(0)) >= 0.05 || std::abs(point(1)) >= 0.05);
}

double getAreaOfTotalCurrent()
{
  return 0.05*0.05;
}

double frequency = 50;

//pro oblast vzduchu :
double ma_gamma_val_outside = 0.;
double ma_mur_val_outside = 1.;

//pro oblast vodice(med) :
double ma_gamma_val_inside = 57.e6;
double ma_mur_val_inside = 1.;

//v prvni iteraci buzene externi proudovou hustotou(med)
#if INCLUDE_TOTAL_CURRENT == 0
double ma_Jer_val = 337.97;
double ma_Jei_val = 2511.17;
#else
double ma_Jer_val = 0.;
double ma_Jei_val = 0.;
#endif

template <int dim>
class HelmoltzProblem
{
public:
  HelmoltzProblem();
  ~HelmoltzProblem();
  void run();

private:
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;

  FESystem<dim>        fe;

  ConstraintMatrix     hanging_node_constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  void process_matrix_total_current(SparseMatrix<double>& system_matrix);
  void process_rhs_total_current(Vector<double>& system_rhs);
};

template <int dim>
HelmoltzProblem<dim>::HelmoltzProblem()
  :
  dof_handler(triangulation),
  fe(FE_Q<dim>(1), dim)
{
}

template <int dim>
HelmoltzProblem<dim>::~HelmoltzProblem()
{
  dof_handler.clear();
}

template <int dim>
void HelmoltzProblem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
    hanging_node_constraints);
  hanging_node_constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
    dsp,
    hanging_node_constraints,
    /*keep_constrained_dofs = */ true);

  sparsity_pattern.copy_from(dsp, INCLUDE_TOTAL_CURRENT * 2);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs() + INCLUDE_TOTAL_CURRENT * 2);
  system_rhs.reinit(dof_handler.n_dofs() + INCLUDE_TOTAL_CURRENT * 2);
}

template <int dim>
void HelmoltzProblem<dim>::assemble_system()
{
  QGauss<dim>  quadrature_formula(2);

  FEValues<dim> fe_values(fe, quadrature_formula,
    update_values | update_gradients |
    update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points = quadrature_formula.size();

  FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int
        component_i = fe.system_to_component_index(i).first;

      if (component_i >= dim)
        continue;

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int
          component_j = fe.system_to_component_index(j).first;

        for (unsigned int q_point = 0; q_point < n_q_points;
          ++q_point)
        {
          const dealii::Point<2> p = fe_values.quadrature_point(q_point);

          double ma_gamma_val;
          double ma_mur_val;
          if (isOutsideOfSmallSquare(p))
          {
            ma_gamma_val = ma_gamma_val_outside;
            ma_mur_val = ma_mur_val_outside;
          }
          else
          {
            ma_gamma_val = ma_gamma_val_inside;
            ma_mur_val = ma_mur_val_inside;
          }

          if (component_i == component_j))
          {
            cell_matrix(i, j) += fe_values.JxW(q_point) *(1. / (ma_mur_val*1.25664e-06)*(fe_values.shape_grad(j, q_point)[0] * fe_values.shape_grad(i, q_point)[0] + fe_values.shape_grad(j, q_point)[1] * fe_values.shape_grad(i, q_point)[1]));
          }
          else if (component_i == 0 && component_j == 1)
          {
            cell_matrix(i, j) += fe_values.JxW(q_point) *(-2. * M_PI*frequency*ma_gamma_val*fe_values.shape_value(j, q_point)*fe_values.shape_value(i, q_point));
          }
          else if (component_i == 1 && component_j == 0)
          {
            cell_matrix(i, j) += fe_values.JxW(q_point) *(2. * M_PI*frequency*ma_gamma_val*fe_values.shape_value(j, q_point)*fe_values.shape_value(i, q_point));
          }
        }
      }
    }

#if INCLUDE_TOTAL_CURRENT == 0
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int
        component_i = fe.system_to_component_index(i).first;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const dealii::Point<2> p = fe_values.quadrature_point(q_point);
        if (!isOutsideOfSmallSquare(p))
        {
          if (component_i == 0)
          {
            cell_rhs(i) += fe_values.JxW(q_point) * ma_Jer_val*fe_values.shape_value(i, q_point);
          }
          else if (component_i == 1)
          {
            cell_rhs(i) += fe_values.JxW(q_point) * ma_Jei_val*fe_values.shape_value(i, q_point);
          }
        }
      }
    }
#endif

    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i],
        local_dof_indices[j],
        cell_matrix(i, j));

      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

#if INCLUDE_TOTAL_CURRENT == 1
  ///// Celkovy proud - custom handling
  this->process_matrix_total_current(system_matrix);
  this->process_rhs_total_current(system_rhs);
#endif

  hanging_node_constraints.condense(system_matrix);
  hanging_node_constraints.condense(system_rhs);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
    0,
    ZeroFunction<dim>(dim + 2 * INCLUDE_TOTAL_CURRENT),
    boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
    system_matrix,
    solution,
    system_rhs);
}

template <int dim>
void HelmoltzProblem<dim>::process_matrix_total_current(SparseMatrix<double>& system_matrix)
{
  QGauss<dim>  quadrature_formula(2);

  FEValues<dim> fe_values(fe, quadrature_formula,
    update_values | update_gradients |
    update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points = quadrature_formula.size();

  FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell != endc; ++cell)
  {
    cell_matrix = 0;
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      const unsigned int
        component_j = fe.system_to_component_index(j).first;

      for (unsigned int q_point = 0; q_point < n_q_points;
        ++q_point)
      {
        const dealii::Point<2> p = fe_values.quadrature_point(q_point);
        if (!isOutsideOfSmallSquare(p))
        {
          if (component_j == 1)
          {
            cell_matrix(0, j) += fe_values.JxW(q_point) *(-2. * M_PI*frequency*ma_gamma_val_inside*fe_values.shape_value(j, q_point));
          }
          else if (component_j == 0)
          {
            cell_matrix(1, j) += fe_values.JxW(q_point) *(2. * M_PI*frequency*ma_gamma_val_inside*fe_values.shape_value(j, q_point));
          }
        }
      }
    }

    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      system_matrix.add(dof_handler.n_dofs(),
        local_dof_indices[j],
        cell_matrix(0, j));

      system_matrix.add(dof_handler.n_dofs() + 1,
        local_dof_indices[j],
        cell_matrix(1, j));
    }

    // Handle the special case - the last columns
    // We do this, because we want to use cell_matrix's special index '0'
    // but if we had done it before, we would be mixing with real values
    cell_matrix = 0;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      const unsigned int
        component_i = fe.system_to_component_index(i).first;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        const dealii::Point<2> p = fe_values.quadrature_point(q_point);
        if (!isOutsideOfSmallSquare(p))
        {
          if (component_i == 0)
          {
            cell_matrix(i, 0) -= fe_values.JxW(q_point) * fe_values.shape_value(i, q_point);
          }
        }
      }
    }
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      system_matrix.add(local_dof_indices[i],
        dof_handler.n_dofs(),
        cell_matrix(i, 0));
    }
  }
  system_matrix.add(dof_handler.n_dofs(),
    dof_handler.n_dofs(),
    getAreaOfTotalCurrent());

  system_matrix.add(dof_handler.n_dofs() + 1,
    dof_handler.n_dofs() + 1,
    getAreaOfTotalCurrent());
}

template <int dim>
void HelmoltzProblem<dim>::process_rhs_total_current(Vector<double>& system_rhs)
{
  system_rhs(dof_handler.n_dofs()) = TOTAL_CURRENT;
}

template <int dim>
void HelmoltzProblem<dim>::solve()
{
  if (PRINT_ALGEBRA)
  {
    std::cout << "  Printing system... " << std::endl;

    std::string matrix_file = "Matrix.txt";
    std::string rhs_file = "Rhs.txt";

    std::ofstream matrix_out(matrix_file);
    std::ofstream rhs_out(rhs_file);

    matrix_out << std::fixed;
    rhs_out << std::fixed;
    matrix_out << std::setprecision(6);
    rhs_out << std::setprecision(6);
    system_matrix.print(matrix_out, false, false);
    system_rhs.print(rhs_out, 6, false, false);

    matrix_out.close();
    rhs_out.close();
  }

  dealii::SparseDirectUMFPACK solver;

  solver.initialize(system_matrix);

  solver.vmult(solution, system_rhs);
}

template <int dim>
void HelmoltzProblem<dim>::output_results() const
{
  std::string filename = "solution";
  filename += ".vtk";

  std::ofstream output(filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;
  switch (dim)
  {
  case 1:
    solution_names.push_back("displacement");
#if INCLUDE_TOTAL_CURRENT == 1
    solution_names.push_back("current_density_R");
    solution_names.push_back("current_density_I");
#endif
    break;
  case 2:
    solution_names.push_back("A_x");
    solution_names.push_back("A_y");
#if INCLUDE_TOTAL_CURRENT == 1
    solution_names.push_back("current_density_R");
    solution_names.push_back("current_density_I");
#endif
    break;
  case 3:
    solution_names.push_back("x_displacement");
    solution_names.push_back("y_displacement");
    solution_names.push_back("z_displacement");
#if INCLUDE_TOTAL_CURRENT == 1
    solution_names.push_back("current_density_R");
    solution_names.push_back("current_density_I");
#endif
    break;
  default:
    Assert(false, ExcNotImplemented());
  }

#if INCLUDE_TOTAL_CURRENT == 1
  std::cout << "Ext_R: " << solution[dof_handler.n_dofs()] << std::endl;
  std::cout << "Ext_I: " << solution[dof_handler.n_dofs() + 1] << std::endl;
#endif

  Vector<double> solution_to_display(dof_handler.n_dofs());
  for (int i = 0; i < dof_handler.n_dofs(); i++)
    solution_to_display[i] = solution[i];

  data_out.add_data_vector(solution_to_display, solution_names);

  data_out.build_patches();
  data_out.write_vtk(output);
}

template <int dim>
void HelmoltzProblem<dim>::run()
{
  GridGenerator::hyper_cube(triangulation, -0.1, 0.1);
  triangulation.refine_global(INIT_REF_NUM);

  std::cout << "   Number of active cells:       "
    << triangulation.n_active_cells()
    << std::endl;

  setup_system();

  std::cout << "   Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;

  assemble_system();
  solve();
  output_results();
}

int main()
{
  try
  {
    dealii::deallog.depth_console(0);

    HelmoltzProblem<2> Helmoltz_problem_2d;
    Helmoltz_problem_2d.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Exception on processing: " << std::endl
      << exc.what() << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
      << "----------------------------------------------------"
      << std::endl;
    std::cerr << "Unknown exception!" << std::endl
      << "Aborting!" << std::endl
      << "----------------------------------------------------"
      << std::endl;
    return 1;
  }

  return 0;
}
