from abc import ABC, abstractmethod

from transformers import (Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor)
from pyctcdecode import build_ctcdecoder

import pickle as pkl
import numpy as np
import librosa
import torch
import os
import shutil

from datasets import load_dataset
from re import sub
from math import ceil
from time import time


class Preprocessing(ABC):
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Preprocessing, cls).__new__(cls)
        return cls._instance
    
    
    def __init__(self, output_folder: str, path_corpus: str, path_model_language: str, device):
        self.device = device
        self.path_corpus = path_corpus
        self.path_model_language = path_model_language
        
        self.output_folder = output_folder + '/wav2vec2_interpretation'
        self.output_data   = self.output_folder + '/data'
        self.output_images = self.output_folder + '/images'
        
        if not Preprocessing._initialized: 
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)  
            
            os.makedirs(self.output_folder)
            os.makedirs(self.output_data)
            os.makedirs(self.output_images)
            
            Preprocessing._initialized = True
            
    
    def freeze_data(self):
        timestamp = str(int(time()))
        os.system(f'mv "{self.output_folder}" "{self.output_folder}_{timestamp}"')
        

    def remove_special_characters(self, batch):
        chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
        batch["sentence"] = batch["sentence"].replace('\n', '')
        batch["sentence"] = sub(chars_to_remove_regex, '', batch["sentence"]).lower()
        return batch
    
    
    def dataset_processor(self, path: str):
        # load transcriptions
        voice  = load_dataset("csv", data_files={"corpus": f"{path}/metadata.csv"})
        # pre-processing
        voice = voice.map(self.remove_special_characters)
        return voice
    
    
    def extract_pre_fine(self, name_model: str, type_model: str):
        model = Wav2Vec2ForCTC.from_pretrained(name_model, cache_dir = "tmp").to(self.device)
        processor =  Wav2Vec2Processor.from_pretrained(name_model, cache_dir = "tmp")

        embeddings = {}

        voices = self.dataset_processor(self.path_corpus)

        for voice in voices['corpus']:
            file_path = voice['path'].split('/')[-1]
            # Loading the audio file
            speech, rate = librosa.load(f"{self.path_corpus}/audios/{file_path}", sr=16000)
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
            input_values_2 = None
            if (input_values.size()[1]/16000 > 300):
                input_values_2 = input_values[:, ceil(input_values.size()[1]/2):]
                input_values = input_values[:, :ceil(input_values.size()[1]/2)]
            with torch.no_grad():
                outputs = model.wav2vec2(input_values.to(self.device)).last_hidden_state

                if input_values_2 != None:
                    outputs_2 = (input_values_2.to(self.device)).last_hidden_state
                    outputs = torch.cat((outputs,outputs_2), 1)

                embeddings[file_path]  = outputs.to("cpu").numpy()

        pkl.dump(embeddings, open(f"{self.output_data}/{type_model}_embeddings.pkl", "wb"))
        
               
    @abstractmethod
    def extract(self, name_model: str):
        pass
    

class Corpus(Preprocessing):
    def __init__(self, output_folder, path_corpus, model_language = None, device = 'cpu'):
        super().__init__(output_folder, path_corpus, model_language, device)  
        

    def extract(self, name_model: str):
        processor = Wav2Vec2Processor.from_pretrained(name_model)
        model_trained = Wav2Vec2ForCTC.from_pretrained(name_model).to(self.device)
        
        decoder = None
        if self.path_model_language is not None:
            labels = processor.tokenizer.convert_ids_to_tokens(
                list(range(processor.tokenizer.vocab_size))
            )
            labels.append("")
            decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=self.path_model_language
            )
        
        transcripts = dict()
        transcripts_with_lm = dict()
        voices = self.dataset_processor(self.path_corpus)

        for voice in voices['corpus']:
            patch_audio = voice['path'].split('/')[-1]
            # Loading the audio file
            audio, rate = librosa.load(f"{self.path_corpus}/audios/{patch_audio}", sr=16000)

            # Getting transcription
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            
            with torch.no_grad():
                logits = model_trained(inputs).logits
            
            if self.path_model_language is not None:
                probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
                
                transcription = decoder.decode(probs)
                predicted_ids_lm = np.array(processor.tokenizer(transcription).input_ids)    
                transcripts_with_lm[f'{self.path_corpus}/audios/{patch_audio}'] = predicted_ids_lm
                
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_ids = np.array(predicted_ids[0].cpu())
            
            transcripts[f'{self.path_corpus}/audios/{patch_audio}'] = predicted_ids

        if self.path_model_language is not None:
            with open(f'{self.output_data}/predicted_ids_lm.pkl', 'wb') as f:
                pkl.dump(transcripts_with_lm, f)
        with open(f'{self.output_data}/predicted_ids.pkl', 'wb') as f:
            pkl.dump(transcripts, f)
    

class CNNEmbeddings(Preprocessing):
    def __init__(self, output_folder, path_corpus, device):
        super().__init__(output_folder, path_corpus, None, device) 


    def extract(self, name_model: str):
        model = Wav2Vec2ForCTC.from_pretrained(name_model, cache_dir = "tmp").to(self.device)
        processor =  Wav2Vec2FeatureExtractor.from_pretrained(name_model, cache_dir = "tmp")
        
        embeddings = dict()

        voices = self.dataset_processor(self.path_corpus)

        for voice in voices['corpus']:
            file_path = voice['path'].split('/')[-1]
            # Loading the audio file
            speech, rate = librosa.load(f"{self.path_corpus}/audios/{file_path}", sr=16000)
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(self.device)

            input_values_2 = None
            if (input_values.size()[1]/16000 > 300):
                input_values_2 = input_values[:, ceil(input_values.size()[1]/2):]
                input_values = input_values[:, :ceil(input_values.size()[1]/2)]
            with torch.no_grad():
                outputs = model.wav2vec2.feature_extractor(input_values).transpose(1, 2)

                if input_values_2 != None:
                    outputs_2 = model.wav2vec2.feature_extractor(input_values).transpose(1, 2)
                    outputs = torch.cat((outputs,outputs_2), 1)

                embeddings[file_path]  = outputs.to("cpu").numpy()

        pkl.dump(embeddings, open(f"{self.output_data}/finetuned_cnn_embeddings.pkl", "wb"))
    

class FinetunedEmbeddings(Preprocessing):
    def __init__(self, output_folder, path_corpus, device):
        super().__init__(output_folder, path_corpus, None, device)  


    def extract(self, name_model: str):
        self.extract_pre_fine(name_model, 'finetuned')


class PretrainedEmbeddings(Preprocessing):
    def __init__(self, output_folder, path_corpus, device):
        super().__init__(output_folder, path_corpus, None, device)  


    def extract(self, name_model: str):
        self.extract_pre_fine(name_model, 'pretrained')
        