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

const bool PRINT_ALGEBRA = true;

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

namespace Step8
{
    using namespace dealii;
    template <int dim>
    class ElasticProblem
    {
    public:
        ElasticProblem();
        ~ElasticProblem();
        void run();

    private:
        void setup_system();
        void assemble_system();
        void solve();
        void refine_grid();
        void output_results() const;

        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dof_handler;

        FESystem<dim>        fe;

        ConstraintMatrix     hanging_node_constraints;

        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;

        Vector<double>       solution;
        Vector<double>       system_rhs;

        unsigned int main_equation_index;
        void set_main_equation_index(DynamicSparsityPattern& dsp);

        void process_matrix_total_current(SparseMatrix<double>& system_matrix);
        void process_rhs_total_current(Vector<double>& system_rhs);
    };

    template <int dim>
    class RightHandSide : public Function < dim >
    {
    public:
        RightHandSide();

        virtual void vector_value(const Point<dim> &p,
            Vector<double>   &values) const;

        virtual void vector_value_list(const std::vector<Point<dim> > &points,
            std::vector<Vector<double> >   &value_list) const;
    };

    template <int dim>
    RightHandSide<dim>::RightHandSide()
        :
        Function<dim>(dim)
    {}

    template <int dim>
    inline
        void RightHandSide<dim>::vector_value(const Point<dim> &p,
        Vector<double>   &values) const
    {
        Point<dim> point_1, point_2;
        point_1(0) = 0.5;
        point_2(0) = -0.5;

        if (((p - point_1).norm_square() < 0.2*0.2) ||
            ((p - point_2).norm_square() < 0.2*0.2))
            values(0) = 1;
        else
            values(0) = 0;

        if (p.norm_square() < 0.2*0.2)
            values(1) = 1;
        else
            values(1) = 0;
    }

    template <int dim>
    void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
        std::vector<Vector<double> >   &value_list) const
    {
        Assert(value_list.size() == points.size(),
            ExcDimensionMismatch(value_list.size(), points.size()));

        const unsigned int n_points = points.size();

        for (unsigned int p = 0; p < n_points; ++p)
            RightHandSide<dim>::vector_value(points[p],
            value_list[p]);
    }

    template <int dim>
    ElasticProblem<dim>::ElasticProblem()
        :
        dof_handler(triangulation),
        fe(FE_Q<dim>(1), dim, FE_DGQ<dim>(0), 1),
        main_equation_index(0)
    {}

    template <int dim>
    ElasticProblem<dim>::~ElasticProblem()
    {
        dof_handler.clear();
    }

    template <int dim>
    void ElasticProblem<dim>::setup_system()
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

        set_main_equation_index(dsp);

        sparsity_pattern.copy_from(dsp);

        system_matrix.reinit(sparsity_pattern);

        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
    }

    template <int dim>
    void ElasticProblem<dim>::assemble_system()
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

        std::vector<double>     lambda_values(n_q_points);
        std::vector<double>     mu_values(n_q_points);

        ConstantFunction<dim> lambda(1.), mu(1.);

        RightHandSide<dim>      right_hand_side;
        std::vector<Vector<double> > rhs_values(n_q_points,
            Vector<double>(dim + 1));

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
        for (; cell != endc; ++cell)
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                rhs_values);

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

                    if (component_j >= dim)
                        continue;

                    for (unsigned int q_point = 0; q_point < n_q_points;
                        ++q_point)
                    {
                        cell_matrix(i, j)
                            +=
                            (
                            (fe_values.shape_grad(i, q_point)[component_i] *
                            fe_values.shape_grad(j, q_point)[component_j] *
                            lambda_values[q_point])
                            +
                            (fe_values.shape_grad(i, q_point)[component_j] *
                            fe_values.shape_grad(j, q_point)[component_i] *
                            mu_values[q_point])
                            +
                            ((component_i == component_j) ?
                            (fe_values.shape_grad(i, q_point) *
                            fe_values.shape_grad(j, q_point) *
                            mu_values[q_point]) :
                            0)
                            )
                            *
                            fe_values.JxW(q_point);
                    }
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int
                    component_i = fe.system_to_component_index(i).first;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                    cell_rhs(i) += fe_values.shape_value(i, q_point) *
                    rhs_values[q_point](component_i) *
                    fe_values.JxW(q_point);
            }

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

        ///// Celkovy proud - custom handling
        this->process_matrix_total_current(system_matrix);
        this->process_rhs_total_current(system_rhs);

        hanging_node_constraints.condense(system_matrix);
        hanging_node_constraints.condense(system_rhs);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
            0,
            ZeroFunction<dim>(dim + 1),
            boundary_values);
        MatrixTools::apply_boundary_values(boundary_values,
            system_matrix,
            solution,
            system_rhs);
    }

    template <int dim>
    void ElasticProblem<dim>::set_main_equation_index(DynamicSparsityPattern& dsp)
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

        std::vector<double>     lambda_values(n_q_points);
        std::vector<double>     mu_values(n_q_points);

        ConstantFunction<dim> lambda(1.), mu(1.);

        RightHandSide<dim>      right_hand_side;
        std::vector<Vector<double> > rhs_values(n_q_points,
            Vector<double>(dim + 1));

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
        for (; cell != endc; ++cell)
        {
            cell_matrix = 0;
            cell_rhs = 0;

            fe_values.reinit(cell);

            lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
            mu.value_list(fe_values.get_quadrature_points(), mu_values);

            cell->get_dof_indices(local_dof_indices);

            right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                rhs_values);

            bool is_this_main_equation = false;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int
                    component_i = fe.system_to_component_index(i).first;

                if (component_i == dim && main_equation_index == 0)
                    main_equation_index = local_dof_indices[i];

                if (component_i == dim && main_equation_index != 0)
                    dsp.add(local_dof_indices[i], main_equation_index);
            }
        }

        for (int i = 0; i < dof_handler.n_dofs(); i++)
        {
            dsp.add(this->main_equation_index, i);
        }
    }

    template <int dim>
    void ElasticProblem<dim>::process_matrix_total_current(SparseMatrix<double>& system_matrix)
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

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int
                    component_i = fe.system_to_component_index(i).first;

                if (component_i < dim)
                    continue;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int
                        component_j = fe.system_to_component_index(j).first;

                    // component_j == 0 -> to je tady simulace nejakeho clenu ktery v realu je
                    // vypocet indukovanych proudu, cili gamma * (v x B)
                    // component_j == dim je pak skutecne to co chceme distribuovat a sice J_{ext}
                    if ((component_j == 0) || (component_j == dim)) {
                        for (unsigned int q_point = 0; q_point < n_q_points;
                            ++q_point)
                        {
                            cell_matrix(i, j)
                                +=
                                fe_values.shape_value(i, q_point) *
                                fe_values.shape_value(j, q_point) *
                                fe_values.JxW(q_point);
                            std::cout << "cell " << cell << " comp " << component_j << " j: " << j << " : " << fe_values.shape_value(i, q_point) << std::endl;
                        }
                    }
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int
                    component_i = fe.system_to_component_index(i).first;

                if (component_i < dim)
                    continue;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    system_matrix.add(main_equation_index,
                        local_dof_indices[j],
                        cell_matrix(i, j));
                }

                if (main_equation_index != local_dof_indices[i])
                {
                    system_matrix.add(local_dof_indices[i],
                        main_equation_index,
                        1.);
                    system_matrix.add(local_dof_indices[i],
                        local_dof_indices[i],
                        -1.);
                }
            }
        }
    }

    template <int dim>
    void ElasticProblem<dim>::process_rhs_total_current(Vector<double>& system_rhs)
    {
        const unsigned int   dofs_per_cell = fe.dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
            endc = dof_handler.end();
        for (; cell != endc; ++cell)
        {
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                if (main_equation_index != local_dof_indices[i])
                    continue;
                else // The following means literaly "5 Amper"
                    system_rhs(local_dof_indices[i]) = 5.;
            }
        }
    }

    template <int dim>
    void ElasticProblem<dim>::solve()
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
    void ElasticProblem<dim>::refine_grid()
    {
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(dof_handler,
            QGauss<dim - 1>(2),
            typename FunctionMap<dim>::type(),
            solution,
            estimated_error_per_cell);

        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
            estimated_error_per_cell,
            0.3, 0.03);

        triangulation.execute_coarsening_and_refinement();
    }

    template <int dim>
    void ElasticProblem<dim>::output_results() const
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
            solution_names.push_back("total_current");
            break;
        case 2:
            solution_names.push_back("x_displacement");
            solution_names.push_back("y_displacement");
            solution_names.push_back("total_current");
            break;
        case 3:
            solution_names.push_back("x_displacement");
            solution_names.push_back("y_displacement");
            solution_names.push_back("z_displacement");
            solution_names.push_back("total_current");
            break;
        default:
            Assert(false, ExcNotImplemented());
        }

        data_out.add_data_vector(solution, solution_names);
        data_out.build_patches();
        data_out.write_vtk(output);
    }

    template <int dim>
    void ElasticProblem<dim>::run()
    {
        GridGenerator::hyper_cube(triangulation, 0., 1.);
        triangulation.refine_global(1);

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
}

int main()
{
    try
    {
        dealii::deallog.depth_console(0);

        Step8::ElasticProblem<2> elastic_problem_2d;
        elastic_problem_2d.run();
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
