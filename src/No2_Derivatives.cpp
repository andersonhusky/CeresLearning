#include<iostream>
#include "ceres/ceres.h"

using namespace std;

struct  NumericDiffCostFunctor{
    bool operator()(const double* const x, double* residual) const{
        residual[0] = 10.0 -x[0];
        return true;
    }
};

class QuadraticCostFunction: public ceres::SizedCostFunction<1, 1>{
public:
    virtual ~QuadraticCostFunction(){}
    virtual bool Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const{
        const double x = parameters[0][0];
        residuals[0] = 10-x;

        if(jacobians != nullptr && jacobians[0] != nullptr){
            jacobians[0][0] = -1;
        }
        return true;
    }
};

int main(){
    double initial_x=5.0;
    double x = initial_x;
    
    ceres::Problem problem;
    // ①
    // 第二项：indicates the kind of finite differencing scheme to be used for computing the numerical derivatives
    ceres::CostFunction* cost_function =
        new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(new NumericDiffCostFunctor);

    // ②write Derivatives by youself
    // ceres::CostFunction* cost_function = new QuadraticCostFunction;

    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;
    cout << "x: " << initial_x << "-> " << x << endl;

    return 0;
}