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
