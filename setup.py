from setuptools import setup, find_packages

setup(
    name="clap",
    version="0.1",
    author="Jian Zhu",
    author_email="jian.zhu@ubc.ca",
    description="A short description of your package",
    url="https://github.com/yourusername/your-package-name",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)