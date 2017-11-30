try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='basalunit',
    version='0.1',
    author='Shailesh Appukuttan',
    author_email='shailesh.appukuttan@unic.cnrs-gif.fr',
    packages=['basalunit',
              'basalunit.capabilities',
              'basalunit.tests',
              'basalunit.scores',
              'basalunit.plots'],
    url='https://github.com/appukuttan-shailesh/basalunit',
    download_url = 'https://github.com/appukuttan-shailesh/basalunit/archive/0.1.tar.gz', 
    keywords = ['basal ganglia', 'electrical', 'efel', 'bluepyopt', 'validation framework'],
    license='MIT',
    description='A SciUnit library for data-driven testing of basal ganglia models.',
    long_description="",
    install_requires=['neo','elephant','sciunit>=0.1.5.2',],
    dependency_links = ['git+http://github.com/neuralensemble/python-neo.git#egg=neo-0.4.0dev',
                        'https://github.com/scidash/sciunit/tarball/dev']
)
