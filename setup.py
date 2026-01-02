"""
Setup script for Tree Canopy Detection (TCD) package.
"""

import glob
import os
from pathlib import Path
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
import torch

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            # Filter out comments and empty lines
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []


def get_extensions():
    """Get C++/CUDA extensions for deformable convolution."""
    this_dir = Path(__file__).parent
    extensions_dir = this_dir / "tcd" / "layers" / "csrc"
    
    main_source = str(extensions_dir / "vision.cpp")
    sources = [str(main_source)]
    
    extension = CppExtension
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []
    
    # Add CUDA sources if available
    if torch.cuda.is_available() and CUDA_HOME is not None:
        source_cuda = glob.glob(str(extensions_dir / "deformable" / "*.cu"))
        sources += source_cuda
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    
    include_dirs = [str(extensions_dir)]
    
    ext_modules = [
        extension(
            "tcd._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    
    return ext_modules


setup(
    name='tcd',
    version='0.1.0',
    description='Tree Canopy Detection - Instance segmentation for tree canopy detection using SatlasPretrain backbone',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/Tree_Canopy_Detection',
    packages=find_packages(exclude=['tests', 'tests.*', 'data', 'data.*', 'pretrained', 'pretrained.*']),
    package_dir={'': '.'},
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='tree canopy detection, instance segmentation, remote sensing, computer vision',
    zip_safe=False,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

