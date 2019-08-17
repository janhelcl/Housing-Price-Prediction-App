from setuptools import setup

setup(
    name='housing_api',
    packages=['housing_api'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)