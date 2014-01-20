AC_PREREQ([2.69])
AC_INIT([facelbp], [0.1.0])
AC_CONFIG_SRCDIR([src/common.h])

AM_INIT_AUTOMAKE([1.11 foreign silent-rules])

AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_HOST

dnl make aclocal work in maintainer mode
AC_SUBST(ACLOCAL_AMFLAGS, "-I m4")

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_LIBTOOL

AC_OPENMP
AC_SUBST(OPENMP_CFLAGS)
dnl OpenMP checker only defines for C when compiling both C and C++
OPENMP_CXXFLAGS=$OPENMP_CFLAGS
AC_SUBST(OPENMP_CXXFLAGS)

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h stdint.h stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T

# check for opencl
AC_ARG_ENABLE(opencl,
[  --enable-opencl         enable OpenCL optimizations (default no)],,[enable_opencl=no])
AS_IF([test "${enable_opencl}" != "no"], [
  case "${host_os}" in
    darwin*)
      OPENCL_LIBS="-framework OpenCL"
      ;;
    *)
      OPENCL_LIBS="-lOpenCL"
      ;;
  esac
  AC_SUBST(OPENCL_LIBS)
])
AM_CONDITIONAL([USE_OPENCL], [test "$enable_opencl" = "yes"])

# check for gtk clutter demo
AC_ARG_ENABLE(gtkdemo,
[  --disable-gtkdemo       disable gtk demo (default auto)],,[
  case "${host_os}" in
    *linux*)
      enable_gtkdemo="yes"
      ;;
    *)
      enable_gtkdemo="no"
      ;;
  esac
])
AS_IF([test "${enable_gtkdemo}" != "no"], [
  PKG_CHECK_MODULES(GTK_GST_VIDEO, gtk+-3.0 clutter-gst-2.0 clutter-gtk-1.0 gstreamer-plugins-base-1.0 gstreamer-base-1.0 gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0,, [
    AC_MSG_WARN([${GTK_GST_VIDEO_PKG_ERRORS}.])
    enable_gtkdemo=no
  ])
])
AM_CONDITIONAL([ENABLE_GTKDEMO], [test "$enable_gtkdemo" = "yes"])

# check for sse2 intrinsics
AC_ARG_ENABLE(sse2,
[  --disable-sse2          disable SSE2 optimizations (default auto)],,[
  case "${host_cpu}" in
    i*86|x86_64|amd64)
      enable_sse2="yes"
      ;;
    *)
      enable_sse2="no"
      ;;
  esac
])
have_sse2="no"
AS_IF([test "${enable_sse2}" != "no"], [
  CFLAGS_save="${CFLAGS}"
  CFLAGS="${CFLAGS} -msse2"
  AC_CACHE_CHECK([if $CC groks SSE2 inline assembly], [ac_cv_sse2_inline], [
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM(,[[
void *p;
asm volatile("punpckhqdq %%xmm1,%%xmm2"::"r"(p):"xmm1", "xmm2");
]])
    ], [
      ac_cv_sse2_inline=yes
    ], [
      ac_cv_sse2_inline=no
    ])
  ])
  CFLAGS="${CFLAGS_save}"
  AS_IF([test "${ac_cv_sse2_inline}" != "no" -a "${SYS}" != "solaris"], [
    AC_DEFINE(CAN_COMPILE_SSE2, 1, [Define to 1 if SSE2 inline assembly is available.])
    have_sse2="yes"
  ])
])
AM_CONDITIONAL([HAVE_SSE2], [test "$have_sse2" = "yes"])

# check for xml library support for convert opencv xml data file to our format
# if missing not affect normal using of the library
have_xml="no"
PKG_CHECK_MODULES(XML, libxml-2.0, have_xml="yes", [AC_MSG_WARN([${XML_PKG_ERRORS}.])])
AM_CONDITIONAL([HAVE_XML], [test "$have_xml" = "yes"])

# Checks for library functions.
AC_FUNC_MALLOC

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 data/Makefile
                 examples/Makefile])
AC_OUTPUT

echo "

Configuration
        GTK Demo                   : ${enable_gtkdemo}
        Using OpenCL               : ${enable_opencl}

face_detect configured. Type 'make' to build.
"
