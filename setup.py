from setuptools import setup, find_packages

setup(
    name="vri",  
    version="0.1.0",  
    # packages=["vri"], 
    packages=find_packages(), 
    install_requires=[
        'flask==3.1.1',
        'imageio==2.37.0',
        'imageio-ffmpeg==0.6.0',
        'opencv-python==4.12.0.88',
        'Pillow==11.3.0',
        'einops==0.8.0',
        'matplotlib==3.10.0',
        'dm_env==1.6',
        'websocket-client==1.9.0',
        'msgpack==1.1.0',
        'pyudev==0.24.3'
    ],
)