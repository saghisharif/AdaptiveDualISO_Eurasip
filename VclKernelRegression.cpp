
#include "VclKernelRegression.h"
#include <iostream>
/*----------------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------------*/

namespace vcl{
  
  void CRK0_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float h)
  {
    // compute 
    
    Eigen::MatrixXf 
      tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + 
			 samplePos.col(1).array().pow(2)))); 
    Eigen::MatrixXf weightsi = weights / weights.sum();
    Eigen::MatrixXf W = (tt.array().array().exp() *
			 weightsi.array()) / (sqrtf(2.f *3.141593f)*h*h);
    float ww  = W.sum();
    //W = W / ww;
    float val = (W.transpose() * data)(0) / ww;
    Eigen::ArrayXf p0 =  
      Eigen::ArrayXf::Constant(samplePos.rows(),1, val);

    
    
    //W.array() = W.array() * W.array();
    //Eigen::ArrayXf var = variances.array()* W.array();
    Eigen::MatrixXf sigma(variances.rows(), variances.rows());
    sigma.setZero();
    sigma = variances.asDiagonal();
    W = W / W.sum();
    W.array() = W.array() * W.array();
    Eigen::ArrayXf dist = (p0 - data.array());
    Eigen::ArrayXf err = (dist * dist)* W.array();

    Eigen::MatrixXf var = (W.transpose() * sigma)/ww ;
    estimate(0,0) = val;
    estimate(3,0) = sqrtf(err.sum());
    estimate(4,0) = var.sum();
    estimate(5,0) = W.sum();
  }

  void GetCRK0Kernel(Eigen::MatrixXf &kernel,
		     Eigen::MatrixXf &samplePos,
		     float h)
  {
    Eigen::MatrixXf 
      tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + 
			 samplePos.col(1).array().pow(2)).cast<float>()));
    Eigen::MatrixXf W = (tt.array().exp()) / (sqrtf(2.f *3.141593f)*h*h);
    float ww = W.col(0).array().sum();
    kernel = W.transpose() / ww;
  }

  int GetNumberOfValidSamples(Eigen::MatrixXf &weights,
			     Eigen::MatrixXf &samplePos,
			     float h,
			     float t)
  {
    Eigen::MatrixXf 
      tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + 
			 samplePos.col(1).array().pow(2)).cast<float>()));
    Eigen::MatrixXf W = (tt.array().exp()) / (sqrtf(2.f *3.141593f)*h*h);
    W = W / W.sum();
    Eigen::ArrayXf kernel = weights.array() * W.array();
    int count = 0; 
    //    std::cout << "[ ";
    for(int i = 0; i < kernel.size(); i++){
      //  std::cout << kernel(i) << std::endl;
      if(kernel(i) > t) count ++;
    }
    // std::cout << "]" << std::endl;
    return count;
  }


  void CRK1_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float h){
 
    Eigen::MatrixXf tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + samplePos.col(1).array().pow(2)).cast<float>()));
    tt.array() = tt.array().exp()  / (sqrtf(2.f *3.141593f)*h*h);
   
    Eigen::MatrixXf W = (tt.array()) * weights.array();
   
    // Equivalent kernel
    Eigen::MatrixXf Xx(samplePos.rows(), 3);
    Xx.col(0).setOnes();
    Xx.col(1) = samplePos.col(0);
    Xx.col(2) = samplePos.col(1);

    Eigen::MatrixXf Xw(samplePos.rows(), 3);
    Xw.col(0) = W;
    Xw.col(1) = samplePos.col(0).array() * W.array();
    Xw.col(2) = samplePos.col(1).array() * W.array();
    
    Eigen::MatrixXf A = Xx.transpose() * Xw;
    // solve for coefficients
    Eigen::VectorXf y = (Xw.transpose() * data);
    //Eigen::MatrixXf coeff = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
   

    //W.array() = W.array() * W.array();
   
    //Eigen::ArrayXf var = variances.array()* W.array();
    

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf var(samplePos.rows(),samplePos.rows());

    for(int i=0;i< s.size();i++)
    {

      if(fabs(s(i))<1e-10f)
        s(i)== 1e-8f;

    }
    s = (1.0/s.array()).matrix();
    Eigen::MatrixXf S(s.size(), s.size());
    S.setZero();
    S = s.asDiagonal();
    Eigen::MatrixXf A_inv = svd.matrixV() * S * svd.matrixU().transpose();

    Eigen::MatrixXf coeff = A_inv * y;

    if(svd.rank() < s.size())
       var(0,0) = 1e12f;
    else
    {

      Eigen::MatrixXf sigma(variances.rows(), variances.rows());
      sigma.setZero();
      sigma = variances.asDiagonal();
      var = A_inv * Xw.transpose()* sigma *  Xw * A_inv.transpose();
    }

      W = W / W.sum();
    estimate(0,0) = coeff(0,0);
    estimate(1,0) = coeff(1,0);
    estimate(2,0) = coeff(2,0);

    Eigen::ArrayXf plane =  
      Eigen::ArrayXf::Constant(samplePos.rows(),1, coeff(0,0)) +
      coeff(1,0) * samplePos.col(0).array() + 
      coeff(2,0) * samplePos.col(1).array();
    W.array() = W.array() * W.array();  
    Eigen::ArrayXf dist = (plane - data.array());
    Eigen::ArrayXf err = (dist * dist)* W.array();
    //Eigen::ArrayXf err = dist.abs()  * W.array();


    estimate(3,0) = sqrtf(err.sum());
    estimate(4,0) = var(0,0);
    estimate(5,0) = W.sum();
  }



  void CRK2_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float T,
		     const float h){

    Eigen::MatrixXf 
      tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + 
			 samplePos.col(1).array().pow(2))));
    tt.array() = tt.array().exp()  / (sqrtf(2.f *3.141593f)*h*h);
  
   
    Eigen::MatrixXf W = (tt.array()) * weights.array();
   
    // Equivalent kernel
    Eigen::MatrixXf Xx(samplePos.rows(), 6);
    Xx.col(0).setOnes();
    Xx.col(1) = samplePos.col(0);
    Xx.col(2) = samplePos.col(1);
    Xx.col(3) = samplePos.col(0).array().pow(2);
    Xx.col(4) = samplePos.col(0).array()* samplePos.col(1).array();
    Xx.col(5) = samplePos.col(1).array().pow(2);

    Eigen::MatrixXf Xw(samplePos.rows(), 6);
    Xw.col(0) = W;
    Xw.col(1) = Xx.col(1).array() * W.array();
    Xw.col(2) = Xx.col(2).array() * W.array();
    Xw.col(3) = Xx.col(3).array() * W.array();
    Xw.col(4) = Xx.col(4).array() * W.array();
    Xw.col(5) = Xx.col(5).array() * W.array();
    
    Eigen::MatrixXf A = Xx.transpose() * Xw;
    // solve for coefficients
    Eigen::VectorXf y = (Xw.transpose()) * data;
    //Eigen::MatrixXf coeff = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::MatrixXf var(samplePos.rows(),samplePos.rows());

    for(int i=0;i< s.size();i++)
    {

      if(fabs(s(i))<1e-10f)
        s(i)== 1e-8f;

    }
    s = (1.0/s.array()).matrix();
    Eigen::MatrixXf S(s.size(), s.size());
    S.setZero();
    S = s.asDiagonal();
   
    Eigen::MatrixXf A_inv = svd.matrixV() * S * svd.matrixU().transpose();

    Eigen::MatrixXf coeff = A_inv * y;

   // if(svd.rank() < s.size())
    //   var(0,0) = 1e12f;
    //else
    //{

      Eigen::MatrixXf sigma(variances.rows(), variances.rows());
      sigma.setZero();
      sigma = variances.asDiagonal();
      
       var = A_inv * Xw.transpose()* sigma *  Xw * A_inv.transpose();
     
    //}
    W = W / W.sum();
    //W.array() = W.array() * W.array();
    //Eigen::ArrayXf var_EVS = variances.array()* W.array();   
    W.array() = W.array() * W.array();
    estimate(0,0) = coeff(0,0);
    estimate(1,0) = coeff(1,0);
    estimate(2,0) = coeff(2,0);
    Eigen::ArrayXf plane =  
      Eigen::ArrayXf::Constant(samplePos.rows(),1, coeff(0,0)) +
      coeff(1,0) * Xx.col(1).array() + 
      coeff(2,0) * Xx.col(2).array() +
      coeff(3,0) * Xx.col(3).array() + 
      coeff(4,0) * Xx.col(4).array() +
      coeff(5,0) * Xx.col(4).array();
    /*Eigen::ArrayXf plane =  
      Eigen::ArrayXf::Constant(samplePos.rows(),1, coeff(0,0)) +
      coeff(1,0) * samplePos.col(0).array() + 
      coeff(2,0) * samplePos.col(1).array();  */
    Eigen::ArrayXf dist = (plane - data.array()) ;// * W.array();
    Eigen::ArrayXf err = (dist * dist)  * W.array();
    //Eigen::ArrayXf err = (plane - data.array()).abs()  * W.array();

    estimate(3,0) = sqrtf(err.sum()); //err.sum();
    estimate(4,0) = var(0,0);
    estimate(5,0) = W.sum();
    estimate(6,0) = var(0,0);
   
  }


void GetCRK1Kernel(Eigen::MatrixXf &kernel,
		   Eigen::MatrixXf &samplePos,
		   float h)
  {
    Eigen::MatrixXf 
      tt((-0.5f/(h*h))*((samplePos.col(0).array().pow(2)  + 
			 samplePos.col(1).array().pow(2)).cast<float>()));
    Eigen::MatrixXf W = (tt.array().exp()) / (sqrtf(2.f *3.141593f)*h*h);
    float ww = W.sum();
    kernel = W / ww;
  }

}; //namepsace vcl
