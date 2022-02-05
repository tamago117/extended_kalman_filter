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

class particleFilter
{
public:
    particleFilter();
    ~particleFilter(){};

    void example();

    void localization(const Eigen::MatrixXd& u,
                      const std::vector<Eigen::MatrixXd>& z,
                      Eigen::MatrixXd& xEst,
                      Eigen::MatrixXd& PEst,
                      Eigen::MatrixXd& px,
                      Eigen::MatrixXd& pw);

private:
    //estimation parameter
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    //simulation parameter
    Eigen::MatrixXd Qsim;
    Eigen::MatrixXd Rsim;

    const double DT = 0.1;
    const double SIM_TIME = 30.0;
    const double MAX_RANGE = 20.0;

    //particle filter parameter
    const int NP = 200;
    const int NTh = NP/2.0;

    bool animation = true;

    //function
    void plotCovarianceEllipse(const Eigen::MatrixXd& xEst, const Eigen::MatrixXd& PEst);

    void resampling(Eigen::MatrixXd& px, Eigen::MatrixXd& pw);

    Eigen::MatrixXd calcCovariance(const Eigen::MatrixXd& xEst,
                                   const Eigen::MatrixXd& px,
                                   const Eigen::MatrixXd& pw);

    double gaussLikelihood(double x, double sigma);

    void observation(const Eigen::MatrixXd& u,
                     const std::vector<Eigen::MatrixXd>& rf_id,
                     Eigen::MatrixXd& xTrue,
                     std::vector<Eigen::MatrixXd>& z,
                     Eigen::MatrixXd& xd,
                     Eigen::MatrixXd& ud);
    
    Eigen::MatrixXd motion_model(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u);

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

    Eigen::MatrixXd cumsum(const Eigen::MatrixXd& a);

};

particleFilter::particleFilter()
{
    srand((unsigned int)time(NULL));

    Q = Eigen::MatrixXd(1, 1);
    Q << std::pow(0.1, 2);

    R = Eigen::MatrixXd(2, 2);
    R << std::pow(0.5, 2), 0,
         0, std::pow(90.0*M_PI/180.0, 2);

    Qsim = Eigen::MatrixXd(1, 1);
    Qsim << std::pow(0.1, 2);

    Rsim = Eigen::MatrixXd(2, 2);
    Rsim << std::pow(0.5, 2), 0,
         0, std::pow(90.0*M_PI/180.0, 2);


}

void particleFilter::example()
{
    double time = 0.0;
    double v, w;

    // RFID positions [x, y]
    std::vector<Eigen::MatrixXd> RFID;
    Eigen::MatrixXd RFID_i(2, 1);
    RFID_i << 10.0, 0.0;
    RFID.emplace_back(RFID_i);
    RFID_i << 10.0, 10.0;
    RFID.emplace_back(RFID_i);
    RFID_i << 0.0, 15.0;
    RFID.emplace_back(RFID_i);

    // State Vector [x y yaw v]'
    Eigen::MatrixXd xEst = Eigen::MatrixXd::Zero(4, 1);
    Eigen::MatrixXd xTrue = Eigen::MatrixXd::Zero(4, 1);
    Eigen::MatrixXd PEst = Eigen::MatrixXd::Identity(4, 4);

    Eigen::MatrixXd px = Eigen::MatrixXd::Zero(4, NP); // Particle store
    Eigen::MatrixXd pw = Eigen::MatrixXd::Zero(1, NP);
    pw.fill(1.0 / NP); // Particle weight
    Eigen::MatrixXd xDR = Eigen::MatrixXd::Zero(4, 1);  // Dead reckoning

    // history
    std::vector<Eigen::MatrixXd> hxEst;
    hxEst.push_back(xEst);
    std::vector<Eigen::MatrixXd> hxTrue;
    hxTrue.push_back(xTrue);
    std::vector<Eigen::MatrixXd> hxDR;
    hxDR.push_back(xTrue);

    while(SIM_TIME >= time)
    {
        time += DT;
        v = 1.0;
        w = 0.7 * sin(time);
        Eigen::MatrixXd u = calc_input(v, w);

        Eigen::MatrixXd ud;
        std::vector<Eigen::MatrixXd> z;
        observation(u, RFID, xTrue, z, xDR, ud);
        localization(u, z, xEst, PEst, px, pw);

        //store data history
        hxEst.push_back(xEst);
        hxDR.push_back(xDR);
        hxTrue.push_back(xTrue);

        if(animation){
            matplotlibcpp::clf();

            for (int i=0; i<z.size(); ++i) {
                std::vector<float> xz, yz;
                xz.push_back(xTrue(0, 0));
                yz.push_back(xTrue(1, 0));
                xz.push_back(z[i](1, 0));
                yz.push_back(z[i](2, 0));
                matplotlibcpp::plot(xz, yz, "-g");
            }

            //particle
            std::vector<float> p_x, p_y;
            for(int i = 0; i <NP; ++i){
                p_x.push_back(px(0, i));
                p_y.push_back(px(1, i));
            }
            matplotlibcpp::plot(p_x, p_y, "*r");

            for (int i=0; i<RFID.size(); ++i) {
                std::vector<float> xz, yz;
                xz.push_back(RFID[i](0, 0));
                yz.push_back(RFID[i](1, 0));
                matplotlibcpp::plot(xz, yz, "*k");
            }

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

            matplotlibcpp::axis("equal");
            matplotlibcpp::grid(true);
            matplotlibcpp::pause(0.001);
        }

    }
}

void particleFilter::localization(const Eigen::MatrixXd& u, //input
                      const std::vector<Eigen::MatrixXd>& z, //noise observation and rf_id position
                      Eigen::MatrixXd& xEst, //state estimation
                      Eigen::MatrixXd& PEst, //covariance
                      Eigen::MatrixXd& px, //particle state
                      Eigen::MatrixXd& pw) //particle weight
{
    for(int i = 0; i < NP; ++i){
        Eigen::MatrixXd x = px.col(i); //state
        double w = pw(0, i); //weight

        //predict with random input sampling
        double ud1 = u(0,0) + randn()*Rsim(0,0);
        double ud2 = u(1,0) + randn()*Rsim(1,1);
        Eigen::MatrixXd ud(2,1);
        ud << ud1, ud2;
        x = motion_model(x, ud);

        //calc importance weight
        for(int j=0; j<z.size(); ++j){
            double dx = x(0,0) - z[j](1,0);
            double dy = x(1,0) - z[j](2,0);
            double prez = sqrt(std::pow(dx, 2) + std::pow(dy, 2));
            double dz = prez - z[j](0,0);

            w = w * gaussLikelihood(dz, sqrt(Q(0,0)));
        }

        //update particle
        px.col(i) = x;
        pw(0,i) = w;
    }

    pw = pw / pw.sum(); //normalize

    xEst = px * pw.transpose();
    PEst = calcCovariance(xEst, px, pw);

    resampling(px, pw);
}

void particleFilter::plotCovarianceEllipse(const Eigen::MatrixXd& xEst, const Eigen::MatrixXd& PEst)
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
    R <<    cos(angle), sin(angle),
            -sin(angle), cos(angle);
    Eigen::MatrixXd fx = R * xy;

    std::vector<float> Px_fx, Py_fx;
    for (int i = 0; i < fx.cols(); i++) {
        Px_fx.push_back(fx(0, i) + xEst(0, 0));
        Py_fx.push_back(fx(1, i) + xEst(1, 0));
    }
    matplotlibcpp::plot(Px_fx, Py_fx, "--g");
}

void particleFilter::resampling(Eigen::MatrixXd& px, Eigen::MatrixXd& pw)
{
    int Neff = 1.0 / (pw*pw.transpose())(0,0);
    if(Neff < NTh){
        Eigen::MatrixXd wcum = cumsum(pw);
        Eigen::MatrixXd base = pw;
        base.fill(1.0/NP);
        Eigen::MatrixXd resample_id = base + Eigen::MatrixXd::Random(base.rows(), base.cols()) / NP;

        std::vector<int> inds;
        int ind = 0;
        for(int i = 0; i < NP; ++i){
            while(resample_id(0,i)>wcum(0,ind))
            {
                ind++;
            }
            inds.emplace_back(ind);
        }

        Eigen::MatrixXd t_px = px;
        for(int i = 0; i < NP; ++i){
            px.col(i) = px.col(inds[i]);
        }

        //init weight
        pw = Eigen::MatrixXd::Zero(1, NP);
        pw.fill(1.0/NP);
    }
}

Eigen::MatrixXd particleFilter::calcCovariance(const Eigen::MatrixXd& xEst, //state estimate
                                               const Eigen::MatrixXd& px, //particle state
                                               const Eigen::MatrixXd& pw) //particle weight
{
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(3,3);

    for(int i = 0; i < px.cols(); ++i){ //particle
        Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(3,1);
        for(int j = 0; j < dx.rows(); ++j){ //[x, y, yaw](mean)
                dx(j,0) = px(j,i) - xEst(j,0); //covariance = sigma(x_i - x_mean)(y_i - y_mean)(yaw_i - yaw_mean)/N
        }
        cov += pw(0,i)*dx*dx.transpose();
        //std::cout << "dx: \n" << dx << std::endl;
    }
    cov *= 1.0/(1.0 - (pw*pw.transpose())(0,0));

    //std::cout << "cov: \n" << cov << std::endl;
    return cov;
}

double particleFilter::gaussLikelihood(double x, double sigma)
{
    return (1/std::sqrt(2*M_PI*std::pow(sigma, 2))) * exp(-std::pow(x, 2)/(2*std::pow(sigma, 2)));
}

//consider noise
void particleFilter::observation(const Eigen::MatrixXd& u, //input
                    const std::vector<Eigen::MatrixXd>& rf_id, //object
                    Eigen::MatrixXd& xTrue, //true value
                    std::vector<Eigen::MatrixXd>& z, //noise observation and rf_id position
                    Eigen::MatrixXd& xd, //dead reckoning
                    Eigen::MatrixXd& ud) //noise input
{
    //caluclate next position
    xTrue = motion_model(xTrue, u);

    z.clear();
    for(int i = 0; i < rf_id.size(); ++i){
        double dx = xTrue(0,0) - rf_id[i](0,0);
        double dy = xTrue(1,0) - rf_id[i](1,0);
        double d = sqrt(std::pow(dx, 2) + std::pow(dy, 2));

        if(d <= MAX_RANGE){
            //add noise to observation
            double dn = d + randn() * Qsim(0,0);
            Eigen::MatrixXd zi(3,1);
            zi << dn, rf_id[i](0,0), rf_id[i](1,0);
            z.emplace_back(zi);
        }
    }

    //add noise to input
    double ud1 = u(0,0) + randn() * Rsim(0,0);
    double ud2 = u(1,0) + randn() * Rsim(1,1);
    ud.resize(2, 1);
    ud << ud1, ud2;

    xd = motion_model(xd, ud);
}

Eigen::MatrixXd particleFilter::motion_model(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u)
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

Eigen::Vector2d particleFilter::calc_input(double v, double w)
{
    Eigen::MatrixXd u(2,1);
    u << v, w;

    return u;
}

Eigen::MatrixXd particleFilter::cumsum(const Eigen::MatrixXd& a)
{
    Eigen::MatrixXd p(a.rows(), a.cols());

    double sum = 0;
    if(a.rows() == 1){
        for(int i = 0; i < p.cols(); ++i){
            sum += a(0,i);
            p(0,i) = sum;
        }
        return p;
    }else{
        for(int j = 0; j < p.cols(); ++j){
            sum = 0;
            for(int i = 0; i < p.rows(); ++i){
                sum += a(i,j);
                p(i,j) = sum;
            }
        }

        return p;
    }


}
