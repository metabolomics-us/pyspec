from sys import version_info
from setuptools import setup, find_packages

if version_info.major == 3 and version_info.minor < 6 or \
        version_info.major < 3:
    print('Your Python interpreter must be 3.6 or greater!')
    exit(1)

setup(name='pyspec',
      description='simple command line tool to compute spectra informations from a given list of chromatograms and storing them in a local database',
      url='https://github.com/metabolomics-us/pyspec',
      author='Gert Wohlgemuth',
      author_email='berlinguyinca@gmail.com',
      license='GPLv3',
      packages=find_packages(),
      scripts=[
          "bin/bv_to_msp.py"
      ],
      setup_requires=['pytest-runner'],
      tests_require=[
          'pytest',
          'pytest-watch',
      ],
      install_requires=[
          "pandas",
          "requests",
          "regex"
      ],
      include_package_data=True,
      zip_safe=False,
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Intended Audience :: Science/Research',
      ])
