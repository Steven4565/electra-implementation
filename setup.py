from setuptools import setup, find_packages

setup(
  name='myelectra-pytorch',
  packages=find_packages(),
  version='0.1.0',
  license='MIT',
  description='My ELECTRA implementation - PyTorch',
  author='Your Name',
  author_email='youremail@example.com',
  url='https://github.com/yourgithub/myelectra-pytorch',
  keywords=[
    'transformers',
    'artificial intelligence',
    'pretraining',
    'electra'
  ],
  install_requires=[
    'torch>=1.6.0',
    'transformers>=3.0.2',
    'scipy',
    'sklearn',
    'tqdm',
    'six',
    'numpy'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
) 