from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wav2vec2-interpretation",
    version="0.0.4",
    author="Aline Rodrigues",
    author_email="rodrigues.aline.n@gmail.com",
    description="Investigating Wav2vec2.0 models context representations and the effects of fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[],
    dependency_links=['https://github.com/rodrigues-aline/wav2vec2_interpretation']
)