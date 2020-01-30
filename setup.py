import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDREAM",
    version="0.0.1",
    author="Julian Theis",
    author_email="jul.theis@gmail.com",
    description="Python and PM4Py based implementation of the Decay Replay Mining approach and the Next TrAnsition Prediction algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Julian-Theis/PyDREAM",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="GPL-3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License v3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
)