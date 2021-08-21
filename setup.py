import setuptools

with open("README.md") as f:
    long_description = f.read().rstrip()

with open("digip/VERSION") as f:
    version = f.read().rstrip()

setuptools.setup(
    name="digip",
    version=version,
    license="GPLv3+",
    author="Huidae Cho",
    author_email="grass4u@gmail.com",
    description="Digip is a Python module for digital image processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HuidaeCho/digip",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    package_data={"digip": ["VERSION"]},
)
