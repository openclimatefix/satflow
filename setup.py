from distutils.core import setup

exec(open("satflow/version.py").read())
setup(
    name="satflow",
    version=__version__,
    packages=["satflow", "satflow.data", "satflow.models"],
    url="https://github.com/openclimatefix/satflow",
    license="MIT License",
    company="Open Climate Fix Ltd",
    author="Jacob Bieker",
    author_email="jacob@openclimatefix.org",
    description="Satellite Optical Flow",
)
