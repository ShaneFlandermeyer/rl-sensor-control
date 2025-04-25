from setuptools import setup, find_packages

setup(
    name='rl-sensor-control',
    version='0.0.0',    
    description='RL Sensor Control for my dissertation research',
    url='https://github.com/ShaneFlandermeyer/tdmpc2-jax',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'mot-py @ git+https://github.com/ShaneFlandermeyer/MOTpy.git@develop',
      'flax-gnn @ git+https://github.com/ShaneFlandermeyer/flax-gnn.git@develop',
      'tdmpc2-jax @ git+https://github.com/ShaneFlandermeyer/tdmpc2-jax.git@develop',
      'bmpc-jax @ git+https://github.com/ShaneFlandermeyer/bmpc-jax.git@develop'
    ],

)