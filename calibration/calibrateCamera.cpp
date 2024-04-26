#include <iostream>
#include <dirent.h>

#include <VclRaw.h>
#include <opencv2/opencv.hpp>
#include <vector> 


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

struct image_stats
{
    cv::Mat cv_std;
    cv::Mat cv_mean;
};

std::string get_file_extension(const std::string& FileName)
{
    if(FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".")+1);
    return "";
}


void get_frame_stats_struct(const std::string fname, image_stats &frames_stats, int i_num_imgs)
{
    std::string xml_file            = "cameras.xml";
    struct dirent *pDirent;
    DIR *pDir;
    pDir = opendir (fname.c_str());
    if (pDir == NULL) 
    {
        std::cout << "Cannot open directory" <<  fname.c_str() << std::endl<< std::endl;
        //return 1;
    }

    cv::Mat * cv_raw_image_total = NULL; 
    
    int i_images_counter = 0;
    std::vector< cv::Mat> cv_raw_image; 

    while ((pDirent = readdir(pDir)) != NULL && i_images_counter <i_num_imgs) 
    {

       if(get_file_extension(pDirent->d_name) == "CR2")
        {
            
            std::cout << "reading raw file " << pDirent->d_name << std::endl << std::endl;
            cv::Mat cv_raw_image_tmp; 
            try
            {
                cv_raw_image_tmp = Vcl::loadRawImage((fname + "/"+ pDirent->d_name), xml_file); 
                std::cout << "read raw file " << std::endl << std::endl;
                if(!cv_raw_image_total)
                {
                    cv_raw_image_total = new cv::Mat(cv_raw_image_tmp);
                    //*cv_raw_image_total = cv::Mat::zeros(cv_raw_image_tmp.cols,cv_raw_image_tmp.rows, CV_64F);    
                }
                cv_raw_image.push_back( cv_raw_image_tmp);
                *cv_raw_image_total = *cv_raw_image_total + cv_raw_image_tmp;
                i_images_counter++;

                std::cout << "Counter:  " << i_images_counter << std::endl << std::endl;
            }
            catch(std::exception &e)
            {
                std::cout << "Error while reading RAW image file! " << std::endl << e.what() << std::endl;
               // return false;
            }
        }
           

    }
    closedir (pDir);
    

    *cv_raw_image_total = *cv_raw_image_total / i_images_counter;

    frames_stats.cv_std = cv::Mat((*cv_raw_image_total).rows , (*cv_raw_image_total).cols, (*cv_raw_image_total).type());

    frames_stats.cv_std = cv::Mat::zeros((*cv_raw_image_total).rows , (*cv_raw_image_total).cols,  (*cv_raw_image_total).type());
    //cv::Mat binary = cv::imread("kick.jpg",0);

    //cv::Mat fg;
    //binary.convertTo(fg,CV_32F);
    //fg = fg + 1;
    //cv::log(fg,fg);
    //cv::convertScaleAbs(fg,fg);
    //cv::normalize(fg,fg,0,255,cv::NORM_MINMAX);
    //cv::imshow("a",*cv_raw_image_total);
    //cv::waitKey(0);

    


    cv::Mat tmp;
    

   
    for (int i=0; i < cv_raw_image.size() ; i++)
    {
        //tmp = cv::Mat((*cv_raw_image_total).rows , (*cv_raw_image_total).cols,  (*cv_raw_image_total).type());

        cv::pow((cv_raw_image.at(i) - *cv_raw_image_total), 2, tmp);
        frames_stats.cv_std += tmp;
       

    }
    

    frames_stats.cv_std.convertTo(frames_stats.cv_std,CV_32F);

    cv::sqrt((frames_stats.cv_std / (i_images_counter-1)) , frames_stats.cv_std);

     std::cout << "Passed" <<  std::endl << std::endl;

    frames_stats.cv_mean = *cv_raw_image_total;
    if(cv_raw_image_total)
        delete cv_raw_image_total;

}

    int main (int c, char *v[]) 
    {
        


        int i_saturation_point = 15282;
        std::string s_b_100             = "../calibration/MarkIII/b_100";
        std::string s_b_1600            = "../calibration/MarkIII/b_1600";
        std::string s_b_dual_100_1600   = "../calibration/MarkIII/b_dual_100_1600";
        std::string s_w_100             = "../calibration/MarkIII/w_100";
        std::string s_w_1600            = "../calibration/MarkIII/w_1600";
        std::string s_w_dual_100_1600   = "../calibration/MarkIII/w_dual_100_1600";




        image_stats b_100;
        image_stats b_1600;
        image_stats b_dual_100_1600;
        image_stats w_100;
        image_stats w_1600;
        image_stats w_dual_100_1600;

        if (c < 3) 
        {
            std::cout << "Usage: calibrateCamera <outputdirname> <numberofimages>" << std::endl<< std::endl;
            return 1;
        }

       
        int i_total_images = atoi(v[2]);
        std::string foutputname = v[1];

        get_frame_stats_struct(s_b_100,b_100, i_total_images);

        //shading Correction
        cv::Mat I_100_s = cv::max((w_100.cv_mean - b_100.cv_mean),0.0f) / (i_saturation_point - b_100.cv_mean);
        cv::Mat I_1600_s = cv::max((w_1600.cv_mean - b_1600.cv_mean),0.0f) / (i_saturation_point - b_1600.cv_mean);
        cv::Mat I_dual_100_1600_s = cv::max((w_dual_100_1600.cv_mean - b_dual_100_1600.cv_mean),0.0f) / (i_saturation_point - b_dual_100_1600.cv_mean);

        cv::Mat gain_s = I_100_s / I_dual_100_1600_s;

        


        return 0;
    }