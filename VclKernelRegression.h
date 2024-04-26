#ifndef __VCL_KERNEL_REGRESSION__
#define __VCL_KERNEL_REGRESSION__

#include <eigen3/Eigen/Dense>

/*----------------------------------------------------------------------------*
 *
 *----------------------------------------------------------------------------*/

namespace vcl{


  void CRK0_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float h);

  void GetCRK0Kernel(Eigen::MatrixXf &kernel,
		     Eigen::MatrixXf &samplePos,
		     float h);

	int GetNumberOfValidSamples(Eigen::MatrixXf &weights,
			     Eigen::MatrixXf &samplePos,
			     float h,
			     float t);

  void CRK1_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float h);

  
  
  void CRK2_estimate(Eigen::Matrix<float, 6, 1> &estimate,
		     const Eigen::MatrixXf &data,
		     const Eigen::MatrixXf &samplePos,
		     const Eigen::MatrixXf &variances,
		     const Eigen::MatrixXf &weights,
		     const float T,
		     const float h);
    
  void GetCRK1Kernel(Eigen::MatrixXf &kernel,
		   Eigen::MatrixXf &samplePos,
		   float h);

  
  
}; // namespace vcl


#endif //  __VCL_KERNEL_REGRESSION__
