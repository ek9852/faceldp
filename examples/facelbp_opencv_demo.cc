#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "face_detect.h"

#include <iostream>
#include <fstream>
#include <sstream>

#define MAX_FACES 10
#define DETECT_SKIP 10

using namespace cv;
using namespace std;

int main(int argc, const char *argv[]) {
    int deviceId = 0;
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    int width = 640;
    int height = 480;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    struct face_det *det = face_detector_create(width, height, height/10);
    int frames = 0;

    struct face fa[MAX_FACES];
    static int detected_faces = 0;

    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;

        int numfaces = MAX_FACES;
        if (detected_faces && (frames % DETECT_SKIP)) {
            face_detector_tracking(det, gray.data, fa, detected_faces, &numfaces);
cout << "tracking" << endl;
        } else {
            face_detector_detect(det, gray.data, fa, &numfaces);
cout << "detect" << endl;
        }
        detected_faces = numfaces;
        frames++;
        for(int j=0;j<numfaces;j++) {
            faces.push_back(Rect(fa[j].x, fa[j].y, fa[j].width, fa[j].height));
        }

        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
        }
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
