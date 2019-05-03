# setup.py
from setuptools import setup, find_packages

setup(name='colourizer-keras',
  version='0.1',
  packages=find_packages(),
  description='run keras on gcloud ml-engine',
  author='Eric Hayes',
  author_email='eshayes@princeton.edu',
  license='MIT',
  install_requires=[
      'keras',
      'h5py',
      'scikit-image'
  ],
  zip_safe=False)
