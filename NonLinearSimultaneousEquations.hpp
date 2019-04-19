#ifndef NONLINEARSIMULTANEOUSEQUATIONS_HPP_INCLUDED
#define NONLINEARSIMULTANEOUSEQUATIONS_HPP_INCLUDED

#include "AutomaticDifferentiation.hpp"
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <atomic>
#include <iostream>

namespace NonLinearSimultaneousEquations {

    using namespace AutomaticDifferentiation;

    constexpr unsigned int MAX_NUM_ITERATION=100;
    constexpr double EPSILON=0.001;
    static std::atomic<bool> STOP_FLAG(false);

    bool solve(
        const MatFuncPtr<double>& y_vec,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    bool solve(
        const std::vector<FuncPtr<double>>& y,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);

    bool solve(
        const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& f,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration=MAX_NUM_ITERATION,
        const double epsilon=EPSILON,
        std::atomic<bool>& stop_flag=STOP_FLAG);
}

inline bool NonLinearSimultaneousEquations::solve(
        const MatFuncPtr<double>& y_mat,
        Eigen::VectorXd& x_val,
        const unsigned int max_num_iteration,
        const double epsilon,
        std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    MatFuncPtr<double> jacobian=y_mat.getJacobian(dim); ///  num_y * dim matrix

    for(size_t i=0; !(stop_flag.load())&& i<max_num_iteration; i++){
        /// check whether the norm of y_mat for current x_val be less than epsilon.
        auto y_mat_val=y_mat(x_val);
        double y_mat_val_norm=y_mat_val.norm();

#ifdef DEBUG
        std::cout << "==== iteration #" << i << " ====" << std::endl
                  << "x_val:" << std::endl << x_val << std::endl
                  << "y_mat_val_norm=" << y_mat_val_norm << std::endl;
#endif

        if(y_mat_val_norm<epsilon){
            std::cout << "y_mat_val_norm(" << y_mat_val_norm << ")<epsilon(" << epsilon << ")" << std::endl;
            return true;
        }

        auto jacobian_val=jacobian(x_val);
        /// solve equation: jacobian delta_x = y_mat for x_val
        /// by using of LU Decomposition
        Eigen::FullPivLU lu=jacobian_val.fullPivLu();
        lu.setThreshold(epsilon);
        auto delta_x=lu.solve(-y_mat_val);

        /// check whether the norm of delta_x be less than epsilon.
        if(delta_x.norm()<epsilon){
            std::cout << "delta_x.norm()<epsilon" << std::endl;
            return true;
        }

        /// update x_val
        x_val += delta_x;
    }
    std::cout << "Not converge iterative calculation!" << std::endl;
    return false;
}


inline bool NonLinearSimultaneousEquations::solve(
    const std::vector<FuncPtr<double>>& y,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    const size_t num_y=y.size();
    MatFuncPtr<double> y_mat(num_y);
    for(size_t i=0; i<num_y; i++){
        y_mat(i)=y[i];
    }
    return solve(y_mat,x_val,max_num_iteration,epsilon,stop_flag);
}

inline bool NonLinearSimultaneousEquations::solve(
    const std::vector<std::function<FuncPtr<double>(const std::vector<FuncPtr<double>>&)>>& f,
    Eigen::VectorXd& x_val,
    const unsigned int max_num_iteration,
    const double epsilon,
    std::atomic<bool>& stop_flag)
{
    size_t dim=x_val.rows();
    std::vector<FuncPtr<double>> x=createVariables<double>(dim);
    size_t num_equations=f.size();
    std::vector<FuncPtr<double>> y;
    for(size_t i=0; i<num_equations; i++){
        y.push_back(std::move(f[i](x)));
    }
    return solve(y,x_val,max_num_iteration,epsilon,stop_flag);
}


#endif // NONLINEARSIMULTANEOUSEQUATIONS_HPP_INCLUDED
