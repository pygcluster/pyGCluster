#!/usr/bin/env python

from distutils.core import setup
setup(name='pyGCluster',
      version='0.18.4',
      py_modules = ['pyGCluster'],
      package_dir = {'': 'src'},
      requires = ['numpy', 'scipy'],
      description='hierachical clustering',
      long_description='pyGCluster',
      author='D. Jaeger, J. Barth, A. Niehues, C. Fufezan',
      author_email='christian@fufezan.net',
      url='http://pyGCluster.github.com',
      license='MIT',
      platforms='any that supports python 2.7',
      classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: POSIX :: SunOS/Solaris',
            'Operating System :: Unix',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'Topic :: Scientific/Engineering :: Education',
            'Topic :: Software Development :: Libraries :: Python Modules'
            ],
      )


