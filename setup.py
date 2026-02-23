from setuptools import setup, find_packages

setup(
    name='papertrades',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'click',
        'requests',
        'pandas',
        'numpy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'papertrades=papertrades.cli:main',
        ],
    },
)
