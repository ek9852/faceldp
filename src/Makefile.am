lib_LTLIBRARIES = libfacelbp.la
libfacelbp_la_CXXFLAGS	= @OPENMP_CXXFLAGS@ -O3 -DDATADIR=\"$(datadir)/@PACKAGE@\"
libfacelbp_la_LDFLAGS	= @OPENMP_CXXFLAGS@
libfacelbp_la_SOURCES = face_detect.cc group_rectangle.cc integral_image.cc lbp_detect.cc

if USE_OPENCL
libfacelbp_la_CXXFLAGS	+= -DUSE_OPENCL
libfacelbp_la_LDFLAGS	+= @OPENCL_LIBS@
libfacelbp_la_SOURCES	+= lbp_cl.cc
endif

if HAVE_SSE2
libfacelbp_la_SOURCES += lbp_detect_sse2.cc
endif

# opencv xml converter
if HAVE_XML
bin_PROGRAMS = facelbp_xml_converter

facelbp_xml_converter_SOURCES = facelbp_xml_converter.cc
facelbp_xml_converter_LDADD = @XML_LIBS@
facelbp_xml_converter_CXXFLAGS = @XML_CFLAGS@
endif

extradir = $(datadir)/@PACKAGE@
extra_DATA = \
	lbp.cl \
	lbp_image.cl
