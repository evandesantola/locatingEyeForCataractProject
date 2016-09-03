/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Evan DeSantola
 * Copyright (C) 2016 Evan DeSantola
 */

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <iostream> 
#include <string> 
#include <sstream> 

#include <cstring>
#include <cmath>
#include <stdio.h>
#include "flandmark_detector.h"

void detectFaceInImage(IplImage *orig, IplImage* input, CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox, double *landmarks)
{
    // Smallest face size.
    CvSize minFeatureSize = cvSize(40, 40);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;

    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);

    // Detect all the faces in the greyscale image.
    rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2, flags, minFeatureSize);
    nFaces = rects->total;

    double t = (double)cvGetTickCount();
    for (int iface = 0; iface < (rects ? nFaces : 0); ++iface)
    {
        IplImage * converted = cvCloneImage(orig);
        cv::Mat mat_img(converted);


        CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
        
        bbox[0] = r->x;
        bbox[1] = r->y;
        bbox[2] = r->x + r->width;
        bbox[3] = r->y + r->height;
        
        flandmark_detect(input, bbox, model, landmarks);

        // display landmarks
        cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
        cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
        cvCircle(orig, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
        for (int i = 2; i < 2*model->data.options.M; i += 2)
        {
	    std::cout<<"First: "<<int(landmarks[i]) <<"\n";

            std::cout<<"Second: "<<int(landmarks[i+1])<<"\n";
           
            
            cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);

        }


            cvCircle(orig, cvPoint(int(landmarks[2]), int(landmarks[2+1])), 3, CV_RGB(0,255,0), CV_FILLED);
            cvCircle(orig, cvPoint(int(landmarks[4]), int(landmarks[4+1])), 3, CV_RGB(0,255,0), CV_FILLED);
            cvCircle(orig, cvPoint(int(landmarks[12]), int(landmarks[12+1])), 3, CV_RGB(0,255,0), CV_FILLED);
            cvCircle(orig, cvPoint(int(landmarks[10]), int(landmarks[10+1])), 3, CV_RGB(0,255,0), CV_FILLED);

         std::stringstream faceSS;
         faceSS << iface;

         //Eye 1 ROI:

         int boxBounds = int((landmarks[12]-landmarks[4])*1.3);

         int boxBoundsY=(boxBounds*2)/3;
         std::cout<<"BoxBonds: " << boxBounds<<"\n";
         std::cout<<"x1: " << landmarks[4]<<"\n";
         std::cout<<"x2: " << landmarks[12]<<"\n";
         int topX  = std::max(0,int(int((landmarks[12] + landmarks[4])/2)-(boxBounds)/2));
         std::cout<<"topX: " << topX<<"\n";
         int topY  = std::max(0,int((landmarks[3])-boxBoundsY/2));

         cv::Rect eye1ROI(topX, topY, boxBounds, (boxBounds*2)/3);
         cv::Mat croppedImage1 =  (cv::Mat(converted))(eye1ROI);
         std::string firstEyeFile = "Face_"+faceSS.str()+ "_Eye1.jpg";
         cv::imwrite(firstEyeFile, croppedImage1);

         //Eye 2 ROI:

         boxBounds = int((landmarks[2]-landmarks[10])*1.3);
         boxBoundsY=(boxBounds*2)/3;
         std::cout<<"BoxBonds: " << boxBounds<<"\n";
         std::cout<<"x1: " << landmarks[2]<<"\n";
         std::cout<<"x2: " << landmarks[10]<<"\n";
         topX  = std::max(0,int(int((landmarks[10] + landmarks[2])/2)-(boxBounds)/2));
         std::cout<<"topX: " << topX<<"\n";
         topY  = std::max(0,int((landmarks[3])-boxBoundsY/2));


         cv::Rect eye2ROI(topX, topY, boxBounds, boxBoundsY);
         cv::Mat croppedImage2 =  (cv::Mat(converted))(eye2ROI);
         std::string secondEyeFile = "Face_"+faceSS.str()+ "_Eye2.jpg";
         cv::imwrite(secondEyeFile, croppedImage2);
    }
    t = (double)cvGetTickCount() - t;
    int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );

    if (nFaces > 0)
    {
        printf("Faces detected: %d; Detection of facial landmark on all faces took %d ms\n", nFaces, ms);
    } else {
        printf("NO Face\n");
    }
    




    cvReleaseMemStorage(&storage);
}

int main( int argc, char** argv ) 
{
    char flandmark_window[] = "flandmark_example1";
    double t;
    int ms;
    
    if (argc < 2)
    {
      fprintf(stderr, "Usage: flandmark_1 <path_to_input_image> [<path_to_output_image>]\n");
      exit(1);
    }
    
    cvNamedWindow(flandmark_window, 0);
    
    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename[] = "haarcascade_frontalface_alt.xml";
    // Load the HaarCascade classifier for face detection.
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
    if( !faceCascade )
    {
        printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }

     // ------------- begin flandmark load model
    t = (double)cvGetTickCount();
    FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");

    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }

    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model
    
    // input image
    IplImage *frame = cvLoadImage(argv[1]);
    if (frame == NULL)
    {
      fprintf(stderr, "Cannot open image %s. Exiting...\n", argv[1]);
      exit(1);
    }
    // convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
    
    int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
    detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks);
    
    cvShowImage(flandmark_window, frame);
    cvWaitKey(0);
    
    if (argc == 3)
    {
      printf("Saving image to file %s...\n", argv[2]);
      cvSaveImage(argv[2], frame);
    }
    
    // cleanup
    free(bbox);
    free(landmarks);
    cvDestroyWindow(flandmark_window);
    cvReleaseImage(&frame);
    cvReleaseImage(&frame_bw);
    cvReleaseHaarClassifierCascade(&faceCascade);
    flandmark_free(model);
}
