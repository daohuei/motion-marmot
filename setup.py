import os
from setuptools import setup, find_packages


def setup_package():
    setup(
        name="motion-marmot",
        description="The marmot serves as a motion detector which can target out all possible motions.",
        url="https://github.com/daohuei/motion-marmot",
        author="daohuei",
        author_email="tingken0214@gmail.com",
        maintainer="daohuei",
        maintainer_email="tingken0214@gmail.com",
        packages=find_packages(exclude=["data", "model"]),
        license="MIT",
        # Note: many of these required packages are included in base python
        # but are listed here because different linux distros use custom
        # python installations.  And users can remove packages at any point
        install_requires=[
            "typer",
            "numpy",
            "pandas",
            "opencv-python",
            "sklearn",
            "visdom"
        ],
    )


if __name__ == "__main__":
    setup_package()
