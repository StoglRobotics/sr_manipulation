from glob import glob
from setuptools import setup, find_packages

package_name = 'sr_moveit2_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='guillaume.walck@stoglrobotics.de',
    description='Package with moveit2 utilities for Python',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
            'scene_manager_node = sr_moveit2_utils.scene_manager_node:main',
            'robot_client_node = sr_moveit2_utils.robot_client_node:main',
            'moveit_client = sr_moveit2_utils.moveit_client:main',
            'scene_manager_client = sr_moveit2_utils.scene_manager_client:main'],
    },
)
