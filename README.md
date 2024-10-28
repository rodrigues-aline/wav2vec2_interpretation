[![forthebadge](https://img.shields.io/static/v1?label=Project&message=Wav2vec2.0_Interpretation&color=orange&leact&logoColor=FFFFFF&labelColor=&style=for-the-badge)]()
[![forthebadge](https://img.shields.io/static/v1?label=Python&message=3.10+&color=blue&logo=Python&logoColor=FFFFFF&labelColor=&style=for-the-badge)](http://www.web2py.com/)


# Wav2vec2.0 Interpretation
Understanding Speech Representation Learning in Wav2Vec 2.0: An Analysis with Dimensionality Reduction and Clustering

- This project aims to refactor the [Wav2vec2Interpretation](https://github.com/aalto-speech/Wav2vec2Interpretation) repository to improve its readability and usability, making it more accessible to new users and contributors. The original repository, created by [aalto-speech](https://github.com/aalto-speech), was designed to investigate the capabilities of pre-trained wav2vec2 models and compare them with fine-tuned models for speech recognition

- This repository investigates the complex speech representations learned by the Wav2Vec 2.0 framework, a state-of-the-art self-supervised model for Automatic Speech Recognition (ASR). We address the challenge of interpreting how this deep learning-based architecture extracts and processes acoustic information using dimensionality reduction and clustering techniques.

### Two Essential Steps:

Wav2Vec 2.0 stands out for its ability to learn robust speech representations without requiring large labeled datasets. This is achieved through its two-stage training process:

- **Pre-training**: The model is exposed to many unlabeled speech data, learning to capture acoustic patterns and build a rich representation space. The CNN layer extracts low-level features from the audio, which are then processed by Transformer layers, generating high-quality contextual representations.
- **Fine-tuning**: With a relatively small labeled dataset, the pre-trained model is fine-tuned for speech recognition, adapting its previously learned representations.

### Analyzing the Representation Space:

This repository focuses on analyzing the representation space learned by Wav2Vec 2.0. By applying dimensionality reduction algorithms, such as PCA, t-SNE and UMAP, we visualize and explore the relationships between speech representations. Subsequently, we employ clustering techniques to identify groups of similar phonetic units, evaluating the model's ability to capture the phonetic structure of the language under study.

- **Objective**: Our objective is to provide insights into the learning process of Wav2Vec 2.0, undertanding how the model organizes acoustic and phonetic information. We investigate the comprehensiveness and quality of the representation space, seeking to understand if the model can generalize to different variants and accents of a specific language. The analysis results can help optimize the model and develop more robust and accurate ASR systems.

### References:
- Original repository: https://github.com/aalto-speech/Wav2vec2Interpretation
- Paper:  [Investigating wav2vec2 context representations and the effects of fine-tuning, a
case-study of a Finnish model ](https://www.isca-archive.org/interspeech_2023/grosz23_interspeech.html)


**Join us on this journey to unravel the mysteries of speech representation learning in Wav2Vec 2.0!**

## Get started 

### Download Package

### Source Code

virtual enviroment instalation (linux - python3)
```
sudo apt install python3.8-venv
```

create enviroment
```
python3 -m venv venv
```

activate enviroment
```
source venv/bin/activate
```

install dependencies
```
pip3 install -r requirements.txt
```
