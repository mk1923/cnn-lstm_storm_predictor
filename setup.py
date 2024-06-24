try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='image_predictor',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Network to predict the next 3 images, given a sequence',
      author='ESE Yolonda group',
      packages=['image_predictor'],
      )

setup(name='wind_predictor',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Network to predict the wind speeds based on given images',
      author='ESE Yolonda group',
      packages=['wind_predictor'],
      )
