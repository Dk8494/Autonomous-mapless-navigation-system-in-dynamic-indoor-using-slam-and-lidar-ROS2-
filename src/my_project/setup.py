from setuptools import find_packages, setup
from glob import glob

package_name = 'my_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name,
         ['package.xml']),
        ('share/' + package_name + '/launch',
         glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf',
         glob('urdf/*')),
        ('share/' + package_name + '/worlds',
         glob('worlds/*')),
         ('share/' + package_name + '/config',
         glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Devendra',
    maintainer_email='devendra@example.com',
    description='SURGE Navigation Project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drl_navigator = my_project.drl_navigator:main',
            'train_agent = my_project.train_agent:main',
        ],
    },
)
