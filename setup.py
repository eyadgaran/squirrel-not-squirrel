from setuptools import setup

setup(name='squirrel',
      version='2.0',
      description='Squirrel-not-Squirrel App',
      url='https://github.com/eyadgaran/squirrel-not-squirrel.git',
      author=['Elisha Yadgaran', 'Justin Su'],
      author_email=['elisha.yadgaran@qpidhealth.com', 'justin.su@qpidhealth.com'],
      license='MIT',
      packages=['squirrel'],
      install_requires=[
          'flask',
          'sqlalchemy',
          'sqlalchemy-mixins',
          'simpleml[all]',
          'opencv-python',
          'tqdm',
          'pandas',
          'numpy',
          'requests',
          'keras',
          'imagehash',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      # include_package_data=True,
      package_dir={'squirrel':'squirrel'},
      package_data={'squirrel': ['templates/*', 'static/*']},
    )
