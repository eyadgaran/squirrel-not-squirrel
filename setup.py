from setuptools import setup, find_packages

setup(name='squirrel',
      version='1.0',
      description='Squirrel App',
      url='',
      author=['Elisha Yadgaran', 'Justin Su'],
      author_email=['elisha.yadgaran@qpidhealth.com', 'justin.su@qpidhealth.com'],
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
