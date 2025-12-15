"""
DenseCore setup.py - Pure Python package

The C++ library must be built separately using:
    cd .. && make lib

This setup.py installs only the Python bindings.
"""
import os
from setuptools import setup, find_packages

# Read long description from README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='densecore',
    version='2.0.0',
    author='DenseCore Team',
    author_email='jake@densecore.ai',
    description='High-Performance CPU Inference Engine for LLMs with HuggingFace Integration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Jake-Network/DenseCore',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'huggingface-hub>=0.20.0',
    ],
    extras_require={
        'full': [
            'transformers>=4.35.0',
            'tokenizers>=0.15.0',
            'numpy>=1.20.0',
        ],
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
    },
    package_data={
        'densecore': ['*.so', '*.dylib', '*.dll', 'py.typed'],
    },
    include_package_data=True,
    zip_safe=False,
)
