try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

json_files = package_files('basalunit/tests/somafeat_stim')

setup(
    name='basalunit',
    version='1.2.0',
    author='Shailesh Appukuttan; Pedro García-Rodríguez',
    author_email='shailesh.appukuttan@unic.cnrs-gif.fr',
    packages=['basalunit',
              'basalunit.capabilities',
              'basalunit.tests',
              'basalunit.scores',
              'basalunit.plots'],
    package_data={'basalunit': json_files},
    url='https://github.com/appukuttan-shailesh/basalunit',
    keywords = ['basal ganglia', 'electrical', 'efel', 'bluepyopt', 'validation framework'],
    license='MIT',
    description='A SciUnit library for data-driven testing of basal ganglia models.',
    long_description="",
    install_requires=['neo','elephant','sciunit>=0.2.7', 'tabulate', 'similaritymeasures']
)
