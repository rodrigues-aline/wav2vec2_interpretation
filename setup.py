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
    install_requires=[
        "matplotlib==3.9.2",
        "distinctipy==1.3.4",
        "numba==0.60.0",
        "numpy==1.23.5",
        "umap-learn==0.5.6",
        "librosa==0.10.2.post1",
        "transformers==4.46.0",
        "torch==2.5.0",
        "coclust==0.2.1",
        "datasets==3.0.2",
        "scikit-learn==1.5.2",
        "scipy==1.13.1",
        "CairoSVG==2.7.1"
    ],
    dependency_links=['https://github.com/rodrigues-aline/wav2vec2_interpretation']
)