"""Install PDIoT-ML package"""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml',
    version='1.0.0',
    author='Jonathan Gustafsson Frennert',
    description='PDIoT ML Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/J0HNN7G/pdiot-ml',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
        'yacs',
        'wandb'
    ]
)