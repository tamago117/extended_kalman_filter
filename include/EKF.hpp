/*
* @editor michikuni eguchi
* @reference Forrest-Z/cpp_robotics https://github.com/Forrest-Z/cpp_robotics
* @brief particle filter localization
*
*/

#pragma once
#include <iostream>
#include "Eigen/Dense"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "matplotlibcpp.h"

class EKF
{
public:
    EKF();
    ~EKF(){};

    void example();

    void localization(const Eigen::MatrixXd& u, //input
                      const Eigen::MatrixXd& z, //noise observation
                      Eigen::MatrixXd& xEst, //state estimation
                      Eigen::MatrixXd& PEst); //covariance

private:
    //estimation parameter
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    //simulation parameter
    Eigen::MatrixXd INPUT_NOISE;
    Eigen::MatrixXd GPS_NOISE;

    const double DT = 0.1;
    const double SIM_TIME = 30.0;

    bool animation = true;

    //function
    void plotCovarianceEllipse(const Eigen::MatrixXd& xEst, const Eigen::MatrixXd& PEst);

    void observation(const Eigen::MatrixXd& u, //input
                    Eigen::MatrixXd& xTrue, //true value
                    Eigen::MatrixXd& z, //noise observation
                    Eigen::MatrixXd& xd, //dead reckoning
                    Eigen::MatrixXd ud); //noise inputs)
    
    Eigen::MatrixXd motion_model(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u);

    Eigen::MatrixXd observation_model(const Eigen::MatrixXd& x);
    //Jacobian of Motion Model
    Eigen::MatrixXd jacob_f(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u);

    Eigen::MatrixXd jacob_h();

    Eigen::Vector2d calc_input(double v, double w);

    inline double randu()
    {
        return (double)rand()/RAND_MAX;
    }

    inline double randn2(double mu, double sigma) {
        return mu + (rand()%2 ? -1.0 : 1.0)*sigma*pow(-log(0.99999*randu()), 0.5);
    }

    inline double randn() {
        return randn2(0, 1.0);
    }

};

EKF::EKF()
{
    srand((unsigned int)time(NULL));

    Q = Eigen::MatrixXd(4,4);
    Q << std::pow(0.1, 2), 0, 0, 0, //x
         0, std::pow(0.1, 2), 0, 0, //y
         0, 0, std::pow(10*M_PI/180, 2), 0, //theta(degree)
         0, 0, 0, std::pow(0.1, 2); //velocity

    R = Eigen::MatrixXd(2, 2);
    R << std::pow(1.5, 2), 0,
         0, std::pow(1.5, 2);

    INPUT_NOISE = Eigen::MatrixXd(2, 1);
    INPUT_NOISE << std::pow(0.2, 2),
                   std::pow(50*M_PI/180.0, 2);
    

    GPS_NOISE = Eigen::MatrixXd(2, 1);
    GPS_NOISE << std::pow(1.2, 2),
                 std::pow(1.2, 2);


}

void EKF::example()
{
    double time = 0.0;
    double v, w;

    // State Vector [x y yaw v]'
    Eigen::MatrixXd xEst = Eigen::MatrixXd::Zero(4, 1);
    Eigen::MatrixXd xTrue = Eigen::MatrixXd::Zero(4, 1);
    Eigen::MatrixXd PEst = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(2,1);
    Eigen::MatrixXd ud = Eigen::MatrixXd::Zero(2,1);

    Eigen::MatrixXd xDR = Eigen::MatrixXd::Zero(4, 1);  // Dead reckoning

    // history
    std::vector<Eigen::MatrixXd> hxEst;
    hxEst.push_back(xEst);
    std::vector<Eigen::MatrixXd> hxTrue;
    hxTrue.push_back(xTrue);
    std::vector<Eigen::MatrixXd> hxDR;
    hxDR.push_back(xTrue);
    std::vector<Eigen::MatrixXd> hz;
    hz.push_back(z);

    while(SIM_TIME >= time)
    {
        time += DT;
        v = 1.0;
        w = 0.6 * sin(time);
        Eigen::MatrixXd u = calc_input(v, w);


        observation(u, xTrue, z, xDR, ud);
        localization(u, z, xEst, PEst);

        //store data history
        hxEst.push_back(xEst);
        hxDR.push_back(xDR);
        hxTrue.push_back(xTrue);
        hz.push_back(z);

        if(animation){
            matplotlibcpp::clf();

            std::vector<float> Px_hz, Py_hz;
            for (int i = 0; i < hz.size(); i++) {
                Px_hz.push_back(hz[i](0, 0));
                Py_hz.push_back(hz[i](1, 0));
            }
            matplotlibcpp::plot(Px_hz, Py_hz, ".g");

            /*for (int i=0; i<hz.size(); ++i) {
                std::vector<float> xz, yz;
                xz.push_back(xTrue(0, 0));
                yz.push_back(xTrue(1, 0));
                xz.push_back(hz[i](0, 0));
                yz.push_back(hz[i](1, 0));
                matplotlibcpp::plot(xz, yz, "-g");
            }*/


            std::vector<float> Px_hxTrue, Py_hxTrue;
            for (int i = 0; i < hxTrue.size(); i++) {
                Px_hxTrue.push_back(hxTrue[i](0, 0));
                Py_hxTrue.push_back(hxTrue[i](1, 0));
            }
            matplotlibcpp::plot(Px_hxTrue, Py_hxTrue, "-b");

            std::vector<float> Px_hxDR, Py_hxDR;
            for (int i = 0; i < hxDR.size(); i++) {
                Px_hxDR.push_back(hxDR[i](0, 0));
                Py_hxDR.push_back(hxDR[i](1, 0));
            }
            matplotlibcpp::plot(Px_hxDR, Py_hxDR, "-k");

            std::vector<float> Px_hxEst, Py_hxEst;
            for (int i = 0; i < hxEst.size(); i++) {
                Px_hxEst.push_back(hxEst[i](0, 0));
                Py_hxEst.push_back(hxEst[i](1, 0));
            }
            matplotlibcpp::plot(Px_hxEst, Py_hxEst, "-r");

            plotCovarianceEllipse(xEst, PEst);

            matplotlibcpp::title("EKF");
            matplotlibcpp::axis("equal");
            matplotlibcpp::grid(true);
            matplotlibcpp::pause(0.001);
        }

    }
}

void EKF::localization(const Eigen::MatrixXd& u, //input
                      const Eigen::MatrixXd& z, //noise observation
                      Eigen::MatrixXd& xEst, //state estimation
                      Eigen::MatrixXd& PEst) //covariance
{
    //predict
    Eigen::MatrixXd xPred = motion_model(xEst, u);
    Eigen::MatrixXd jF = jacob_f(xEst, u);
    Eigen::MatrixXd PPred = jF * PEst * jF.transpose() + Q;

    //update
    Eigen::MatrixXd jH = jacob_h();
    Eigen::MatrixXd zPred = observation_model(xPred);
    Eigen::MatrixXd y = z - zPred;
    Eigen::MatrixXd S = jH * PPred * jH.transpose() + R;
    Eigen::FullPivLU<Eigen::MatrixXd> S_(S);
    Eigen::MatrixXd K = PPred *jH.transpose() * S_.inverse();
    xEst = xPred + K*y;
    PEst = (Eigen::MatrixXd::Identity(xEst.size(), xEst.size()) - K * jH) * PPred;
}

void EKF::plotCovarianceEllipse(const Eigen::MatrixXd& xEst, const Eigen::MatrixXd& PEst)
{
    Eigen::MatrixXd Pxy(2,2);
    Pxy << PEst(0,0), PEst(0,1),
           PEst(1,0), PEst(1,1);

    Eigen::EigenSolver<Eigen::MatrixXd> es(Pxy);
    Eigen::MatrixXd eigval = es.eigenvalues().real();
    Eigen::MatrixXd eigvec = es.eigenvectors().real();

    int bigind, smallind;
    if(eigval(0,0) >= eigval(1,0)){
        bigind = 0;
        smallind = 1;
    }else{
        bigind = 1;
        smallind = 0;
    }

    double a = 0.0;
    if(eigval(bigind,0) > 0){
        a = sqrt(eigval(bigind,0));
    }
    double b = 0.0;
    if(eigval(smallind,0) > 0){
        b = sqrt(eigval(smallind,0));
    }

    int xy_num = (2*M_PI + 0.1) / 0.1 + 1;
    Eigen::MatrixXd xy(2, xy_num);
    double it = 0.0;
    for (int i=0; i<xy_num; i++) {
        xy(0, i) = a * cos(it);
        xy(1, i) = b * sin(it);
        it += 0.1;
    }

    double angle = atan2(eigvec(bigind, 1), eigvec(bigind, 0));
    Eigen::MatrixXd R(2, 2);
    R <<    cos(angle), -sin(angle),
            sin(angle), cos(angle);
    Eigen::MatrixXd fx = R * xy;

    std::vector<float> Px_fx, Py_fx;
    for (int i = 0; i < fx.cols(); i++) {
        Px_fx.push_back(fx(0, i) + xEst(0, 0));
        Py_fx.push_back(fx(1, i) + xEst(1, 0));
    }
    matplotlibcpp::plot(Px_fx, Py_fx, "--g");
}

//consider noise
void EKF::observation(const Eigen::MatrixXd& u, //input
                    Eigen::MatrixXd& xTrue, //true value
                    Eigen::MatrixXd& z, //noise observation
                    Eigen::MatrixXd& xd, //dead reckoning
                    Eigen::MatrixXd ud) //noise inputs)
{
    //caluclate next position
    xTrue = motion_model(xTrue, u);

    //add noise to gps x,y
    z = observation_model(xTrue) + randn()*GPS_NOISE;

    //add noise to input
    ud = u + randn()*INPUT_NOISE;

    xd = motion_model(xd, ud);
}

Eigen::MatrixXd EKF::motion_model(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u)
{
    Eigen::MatrixXd F(4,4);
    F << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 0;

    Eigen::MatrixXd B(4,2);
    B << DT*cos(x(2,0)), 0,
         DT*sin(x(2,0)), 0,
         0, DT,
         1, 0;

    //[x
    // y
    // theta
    // v]
    return F*x + B*u;

}

Eigen::MatrixXd EKF::observation_model(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd H(2,4);
    H << 1, 0, 0, 0,
         0, 1, 0, 0;

    return H * x;
}

//Jacobian of Motion Model
Eigen::MatrixXd EKF::jacob_f(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u)
{
    double v = u(0,0);
    double yaw = x(2,0);

    Eigen::MatrixXd jF(4,4);
    jF << 1, 0, -v*sin(yaw)*DT, cos(yaw)*DT,
          0, 1, v*cos(yaw)*DT, sin(yaw)*DT,
          0, 0, 1, 0,
          0, 0, 0, 1;

    return jF;
}

//Jacobian of Observation Model
Eigen::MatrixXd EKF::jacob_h()
{
    Eigen::MatrixXd jH(2,4);
    jH << 1, 0, 0, 0,
         0, 1, 0, 0;

    return jH;
}

Eigen::Vector2d EKF::calc_input(double v, double w)
{
    Eigen::MatrixXd u(2,1);
    u << v, w;

    return u;
}
