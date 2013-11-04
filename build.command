# !/bin/bash

mkdir Documentation

rm -rf Documentation/*
rm -rf Website/dist/*
rm -rf Documentation_src/build/*

# Evoke Sphinx to create html and pdf documentation
cd Documentation_src
make html latexpdf
cd ..

# Copying pdf documentation to Documentation and Website
cp Documentation_src/build/latex/pyGCluster.pdf Documentation/
cp Documentation_src/build/latex/pyGCluster.pdf Website/dist/

# Copying html documentation to Documentation and Website
cp -R Documentation_src/build/html Documentation/html
cp -R Documentation_src/build/html/* Website/

rm -rf dist/*
# Creating Python packages
python setup.py sdist --formats=bztar,gztar,zip
cd dist
tar xvfj *.bz2
cd ..

# Copying packages to Website
if [ ! -d "$Test" ]; then
echo 'Please clone pyGCluster website repository with command git clone https://github.com/pygcluster/pygcluster.github.io.git Website'    
#cp dist/pyGCluster*.zip     Website/dist/pyGCluster.zip
#cp dist/pyGCluster*.tar.bz2 Website/dist/pyGCluster.tar.bz2