from distutils.core import setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

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
    author_email="jacob@openclimatefix.org",
    description="Satellite Optical Flow",
)
