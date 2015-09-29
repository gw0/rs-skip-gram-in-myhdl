#!/bin/bash
# Requirements installation for conll2015st-ex10
#
# Author: gw0 [http://gw.tnode.com/] <gw.2015@tnode.com>
# License: All rights reserved

NAME="(`basename $(realpath ${0%/*})`)"
SRC="venv/src"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'

cd "${0%/*}"
virtualenv --prompt="$NAME" venv || exit 1
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"
sudo() { [ -x "/usr/bin/sudo" ] && /usr/bin/sudo "$@" || "$@"; }

# Prerequisites for theano
sudo aptitude install python-numpy=1:1.8.2-2 python-scipy=0.14.0-2
[ ! -d $SITE_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
[ ! -d $SITE_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# Requirements
pip install six
pip install myhdl
pip install -e git+https://github.com/jandecaluwe/myhdl#egg=myhdl
pushd "$SRC/myhdl"
git remote add cfelton https://github.com/cfelton/myhdl
git pull cfelton mep111_fixbv
popd

echo
echo "Use: . venv/bin/activate"
echo
