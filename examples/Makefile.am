bin_PROGRAMS = facelbp_test

facelbp_test_SOURCES = facelbp_test.cc
facelbp_test_LDADD = ../src/libfacelbp.la
facelbp_test_CXXFLAGS = -I$(srcdir)/../src

if ENABLE_GTKDEMO
bin_PROGRAMS += facelbp_gtk_demo
facelbp_gtk_demo_SOURCES = facelbp_gtk_demo.cc
facelbp_gtk_demo_LDADD = ../src/libfacelbp.la @GTK_GST_VIDEO_LIBS@
facelbp_gtk_demo_CXXFLAGS = -I$(srcdir)/../src @GTK_GST_VIDEO_CFLAGS@
endif

if ENABLE_OPENCVDEMO
bin_PROGRAMS += facelbp_opencv_demo
facelbp_opencv_demo_SOURCES = facelbp_opencv_demo.cc
facelbp_opencv_demo_LDADD = ../src/libfacelbp.la @OPENCV_LIBS@
facelbp_opencv_demo_CXXFLAGS = -I$(srcdir)/../src @OPENCV_CFLAGS@
endif
