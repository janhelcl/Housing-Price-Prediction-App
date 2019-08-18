from setuptools import setup

setup(
    name='housing_api',
    packages=['api'],
    include_package_data=True,
    install_requires=[
        'flask',
        'requests'
    ],
)