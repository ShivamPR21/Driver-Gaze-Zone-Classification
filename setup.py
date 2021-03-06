import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "dgzc",
    version = "0.0.0",
    author = "Shivam Pandey",
    author_email = "pandeyshivam2017robotics@gmail.com",
    description = ("Pytorch implementation for Diver Gaze Zone classification challenge."),
    license = "AGPLv3",
    keywords = "DeepLearning Pytorch Driver-Gaze-Zone Attention CNN",
    url = "https://github.com/ShivamPR21/Driver-Gaze-Zone-Classification.git",
    packages=['dgzc'],
    install_requires=[
    ],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: Alpha",
        "Topic :: Research",
        "License :: OSI Approved :: AGPLv3 License",
    ],
)
