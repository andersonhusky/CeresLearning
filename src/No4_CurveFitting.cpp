#include<iostream>
#include<ceres/ceres.h>
#include<random>
#include<chrono>
#include<vector>
#include<matplotlib-cpp/matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;
double M=0.3, C=0.1, R=0.5;

struct ExponentialResidual{
    ExponentialResidual(double x, double y): x_(x), y_(y){}
    template<typename T>

    bool operator()(const T* const m, const T* const c, T* residual) const{
        residual[0] = y_ - exp(m[0]*x_+c[0]);
        return true;
    }
    
    private:
        const double x_;
        const double y_;
};

void cerateData(double* data, int kNumObservations=20){
    double tmp[kNumObservations];
    vector<double> x(kNumObservations, 0);
    vector<double> y(kNumObservations, 0);
    vector<double> z(kNumObservations, 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> distribution(0.0, R);
    for(int i=0; i<kNumObservations; i++)
        data[2*i] = 0.1*i;
    for(int i=0; i<kNumObservations; i++)
    {
        tmp[i] = exp(M*data[2*i]+C);
        data[2*i+1] = tmp[i] + distribution(generator);
        // cout << data[2*i] << ", " << tmp[i] << ", " << data[2*i+1] << endl;
        x[i] = data[2*i];
        y[i] = tmp[i];
        z[i] = data[2*i+1];
    }
    plt::named_plot("original", x, y, ":");
    plt::named_plot("noise", x, z, ".");
    plt::legend();
    plt::show();
}

void plotResult(const double m, const double c, int kNumObservations){
    vector<double> x(kNumObservations, 0);
    vector<double> y(kNumObservations, 0);
    vector<double> z(kNumObservations, 0);
    for(int i=0; i<kNumObservations; i++)
    {
        x[i] = 0.1*i;
        y[i] = exp(M*x[i]+C);
        z[i] = exp(m*x[i]+c);
    }
    plt::named_plot("original", x, y, ":");
    plt::named_plot("optimized", x, y, ".");
    plt::legend();
    plt::show();
}

int main(){
    double m = 0.0;
    double c = 0.0;
    int kNumObservations = 50;
    double data[2*kNumObservations];
    cerateData(data, kNumObservations);

    ceres::Problem problem;
    for(int i=0; i<kNumObservations; i++)
    {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                new ExponentialResidual(data[2*i], data[2*i+1]));
        problem.AddResidualBlock(cost_function, nullptr, &m, &c);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    cout << "Initial m = " << m
            << ", c = " << c << endl;
    ceres::Solve(options, &problem, &summary);
    cout << "Final m = " << m
        << ", c = " << c << endl;
    
    plotResult(m, c, kNumObservations);
    return 0;
}