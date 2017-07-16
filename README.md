Face detection library using LBP (Local Binary Patterns) without using OpenCV
=============================================================================

This package implements the face detection using Multi-Block LBP (MB-LBP).
The facial data in data folder is generated from OpenCV lbpcascade_frontalface.xml file 
using facelbp_xml_converter.
Same algorithm as used by OpenCV, but optimizated using OpenMP, OpenCL or SSE2 and without the bulky dependency of OpenCV.
A face tracking function face_detector_tracking is also provided for fast tracking of 
face instead of scan thought the whole image again.
Instead of scaling the image, we scale the detector by using integral image.

Build
-----
Using standard autogen.sh, configure, make
configure options:::

    --disable-openmp        do not use OpenMP
    --enable-opencl         enable OpenCL optimizations (default no)
    --disable-gtkdemo       disable gtk demo (default auto)
    --disable-sse2          disable SSE2 optimizations (default auto)

To build with Android:

    export NDK_PROJECT_PATH=.
    ndk-build NDK_APPLICATION_MK=./Application.mk

Dependency
----------
libxml-2.0 for convert OpenCV facial xml to this program facial data format.
If gtk v4l2 video demo is needed then 
gtk+-3.0, clutter-gst-2.0, clutter-gtk-1.0, gstreamer-plugins-base-1.0, gstreamer-base-1.0, gstreamer-1.0, gstreamer-video-1.0, gstreamer-app-1.0 
are required
If opencv video demo is needed than opencv2 is required
Usage
-----
To generate a raw grayscale image using ImageMagick::

    $ convert face.jpg -depth 8 gray:face.raw

Then run::

    $ facelbp_test <image width> <image height> face.raw

For GTK demo with web camera or video file, run::

    $ facelbp_gtk_demo # for using web camera
    $ facelbp_gtk_demo -f <video.mp4> # for testing with a video file

For OpenCV (only use for windowing system) with web camera demo, run::

    $ facelbp_opencv_demo # for using web camera

Performance
-----------
Depends on the stages and data in your frontalface.txt.
But you can get realtime on the default facial data in 640x480 resolution @ 30 fps on a i7 CPU.

Fine Tuning
-----------
Edit face_detect.cc for the lbp_para::

    struct lbp_para {
        float scaling_factor;
        int step_scale_x;
        int step_scale_y;
        float tracking_scale_down;
        float tracking_scale_up;
        float tracking_offset; // width percentage
        float eps;
        int group_threshold;
        int min_face_width;
    };

Accuracy depends on your frontalface.txt data.
You need to use OpenCV and several thousands of facial images to generate a good one.
The default facial data lbpcascade_frontalface.xml used by OpenCV (which also packed in this program) is not very accuracy.

