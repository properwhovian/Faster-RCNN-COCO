from setuptools import setup, find_packages

setup(
    name='object_orientation',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch==1.9.0',
        'transformers==4.25.0',
        'torchvision==0.10.0',
        'pycocotools==2.0.2',
        'pillow==8.4.0',
        'numpy==1.21.2',
        'scikit-learn==0.24.2'
    ],
)
