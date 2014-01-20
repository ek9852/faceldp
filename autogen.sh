#! /bin/sh

# libtoolize on Darwin systems is glibtoolize
(glibtoolize --version) < /dev/null > /dev/null 2>&1 && LIBTOOLIZE=glibtoolize || LIBTOOLIZE=libtoolize

mkdir -p m4
aclocal -I m4/ $ACLOCAL_FLAGS \
&& $LIBTOOLIZE --force --copy \
&& automake --add-missing \
&& autoconf
