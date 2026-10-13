from glob import glob
from setuptools import find_packages, setup

name = 'mars_training_environment'
setup(name=name, version='0.2.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/' + name]),
                  ('share/' + name, ['package.xml', 'README.md']),
                  ('share/' + name + '/config', glob('config/*.json') + glob('config/*.yaml')),
                  ('share/' + name + '/launch', glob('launch/*.py'))],
      package_data={name:['review_ui.html']},
      install_requires=['setuptools', 'numpy'], zip_safe=True,
      maintainer='Ian Benge', maintainer_email='ianbenge37@gmail.com', license='Apache-2.0',
      description='Martian terrain and reproducible synthetic camera datasets',
      entry_points={'console_scripts': [
          'mars_assets = mars_training_environment.asset_cache:main',
          'mars_generate = mars_training_environment.generate:main',
          'mars_capture = mars_training_environment.capture:main',
          'mars_dataset = mars_training_environment.dataset:main',
          'mars_validate = mars_training_environment.validate:main',
          'mars_review = mars_training_environment.review:main',
          'mars_verify_runtime = mars_training_environment.runtime_check:main',
      ]})
