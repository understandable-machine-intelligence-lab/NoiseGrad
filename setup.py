from setuptools import setup, find_packages


setup(
    name="noisegrad",
    version="0.0.1",
    description="A explanation enhancement method.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=["torch==1.13.1", "tqdm==4.64.1"],
    url="http://github.com/understandable-machine-intelligence-lab/NoiseGrad",
    author=(
        """
        Kirill Bykov,
        Anna Hedström,
        Shinichi Nakajima,
        Marina M.-C. Höhne,
        Artem Sereda
        """
    ),
    author_email=(
        """
        kirill.bykov@campus.tu-berlin.de, 
        anna.hedstroem@tu-berlin.de, 
        nakajima@tu-berlin.de, 
        marina.hoehne@tu-berlin.de
        artem.sereda@campus.tu-berlin.de
        """
    ),
    keywords=["explainable ai", "xai", "machine learning", "deep learning"],
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.9",
    include_package_data=True,
)
