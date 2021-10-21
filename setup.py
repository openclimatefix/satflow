from distutils.core import setup
from pathlib import Path

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()

exec(open("satflow/version.py").read())
setup(
    name="satflow",
    version=__version__,
    packages=["satflow", "satflow.data", "satflow.models"],
    url="https://github.com/openclimatefix/satflow",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jacob Bieker",
    install_requires=install_requires,
    long_description=long_description,
    ong_description_content_type="text/markdown",
    author_email="jacob@openclimatefix.org",
    description="Satellite Optical Flow",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
