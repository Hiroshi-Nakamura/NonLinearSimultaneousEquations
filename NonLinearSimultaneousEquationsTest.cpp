#include "NonLinearSimultaneousEquations.hpp"

#include <iostream>

int main(int argc, char** argv){
    /// prepair initial x
    Eigen::VectorXd x_val(2);
    x_val << 0.0, 0.5;

    /// solve
    NonLinearSimultaneousEquations::solve(
        {
            [](const std::vector<AutomaticDifferentiation::FuncPtr<double>> x)
            {
                return x[0]*x[0]+x[1]*x[1]-1.0; /// on unit circle
            },
            [](const std::vector<AutomaticDifferentiation::FuncPtr<double>> x)
            {
                return (x[0]-1.0)*(x[0]-1.0)+(x[1]-1.0)*(x[1]-1.0)-1.0; /// on unit circle centered at (1.0,1.0)
            }
        }, /// equations
        x_val /// variables
    );

    std::cout << "soluition x:" << std::endl << x_val <<std::endl;
}
