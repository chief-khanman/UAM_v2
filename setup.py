from setuptools import setup, find_packages

setup(
    name='UAM_simulator',
    version='0.1',
    author='Aadit-Farhan Khan',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'geopandas',
        'poetry',
        'osmnx',
        'ffmpeg-python',
        'geodatasets',
        'scipy',
        'gymnasium',
        'rasterio',
        # TODO - Review dependencies
    ],
    python_requires='>=3.6',
    test_suite='tests',
    tests_require=['unittest']
)