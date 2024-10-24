from abc import ABC
from os import path
import traceback

from .preprocessing import CNNEmbeddings, PretrainedEmbeddings, FinetunedEmbeddings, Corpus
from .dimension_reduction import DimensionalityReduction
from .clustering import Clustering
from .visualization import Visualization


class Wav2vec2Interpretation(ABC):
    def __init__(self, model_pre: str, model_fine: str, path_vocab: str, path_corpus: str, output_folder: str, device: str = 'cpu'):
        
        def check_str(s: str, parameter: str):
            if s is None or s == "": raise ValueError(f'parameter ({parameter}) is invalid')
            if s.isdigit(): raise ValueError(f'parameter ({parameter}) is invalid')
            return s
        
        
        def role_corpus():
            print('\n')
            print('CORPUS DESCRIPTION:')
            print('')
            print('-> It is recommended that the corpus be the test dataset of the trained model or a data set that the model has never seen before')
            print('________________________________________________________________________________________________________________________________')
            print('\n')
            print('-> Corpus format:    ')
            print('__ corpus')
            print('______audios')
            print('__________<name-audio-1>.wav')
            print('__________<name-audio-2>.wav')
            print('__________<name-audio-3>.wav')
            print('__________...')
            print('______metadata.csv')
            print('\n')
            print('-> File metadata.csv columns: (sentence, path)')
            print('____sentence: text transcription')
            print('____path:     path audio file (/<name-audio-1>.wav)')
            print('\n')
            
        
        self.model_pre = check_str(model_pre.strip(), 'model_pre')
        self.model_fine = check_str(model_fine.strip(), 'model_pre')
        
        self.path_vocab = check_str(path_vocab.strip(), 'path_vocab')
        self.path_corpus = check_str(path_corpus.strip(), 'path_corpus')
        self.output_folder = check_str(output_folder.strip(), 'output_folder')
        self.device = check_str(device.strip(), 'output_folder')
        
        if not path.isfile(self.path_vocab):
            raise ValueError(f"(path_vocab) not found")
        
        if not path.isdir(self.path_corpus):
            role_corpus()
            raise ValueError(f'(path_corpus) not found')
        if not path.isdir(self.path_corpus+'/audios'):
            role_corpus()
            raise ValueError(f"(path_corpus) not found 'audios' folder")
        if not path.isfile(self.path_corpus+'/metadata.csv'):
            role_corpus()
            raise ValueError(f"(path_corpus) not found 'metadata.csv' file")
        
        if self.device not in ['cuda', 'cpu']:
            raise ValueError(f"(device) requires 'cuda' or 'cpu'")
        
        # preprocessing
        self.corpus = Corpus(self.output_folder, self.path_corpus, self.device)
        self.cnn_embed = CNNEmbeddings(self.output_folder, self.path_corpus, self.device)
        self.pre_embed = PretrainedEmbeddings(self.output_folder, self.path_corpus, self.device)
        self.fine_embed = FinetunedEmbeddings(self.output_folder, self.path_corpus, self.device)
        
        self.dr = None
        self.clusters = None
        self.plot = None
        
    def preprocessing(self):    
        print (f"---> Starting preprocessing on device='{self.device}'")   
        
        print (f"\t 1- Corpus (predicts_ids)...") 
        self.corpus.extract(self.model_fine)
        print (f"\t 2- CNN embeddings...") 
        self.cnn_embed.extract(self.model_fine)
        print (f"\t 3- Pretrained transformers layers...") 
        self.pre_embed.extract(self.model_pre)
        print (f"\t 4- Finetuned transformers layers...") 
        self.fine_embed.extract(self.model_fine)                
        
        self.dr = DimensionalityReduction(self.output_folder)

        
    def dimension_reduction(self, type: str = 'all'):
        # require preprocessing     
        if self.dr is not None:
            print (f"---> Starting dimensionality reduction on device='{self.device}'")   
            if type == 'pca':
                print (f"\t PCA...") 
                self.dr.extract_pca('cnn')
                self.dr.extract_pca('pre')
                self.dr.extract_pca('fine')
            elif type == 'tsne':
                print (f"\t t-SNE...") 
                self.dr.extract_tsne('cnn')
                self.dr.extract_tsne('pre')
                self.dr.extract_tsne('fine')
            elif type == 'umap':
                print (f"\t UMAP...")
                self.dr.extract_umap('cnn')
                self.dr.extract_umap('pre')
                self.dr.extract_umap('fine')
            elif type == 'all':
                print (f"\t CNN embeddings (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('cnn')
                print (f"\t Pretrained transformers layers (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('pre')
                print (f"\t Finetuned transformers layers (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('fine')
            else:
                raise ValueError("(type) parameter is invalid require: 'cnn', 'pre', 'fine' or 'all'.")
            
            self.clusters = Clustering(self.output_folder, self.path_vocab)
            self.plot = Visualization(self.output_folder, self.path_vocab)
        else:
            raise ValueError("(dimension_reduction) function requires data preprocessing")
    
    
    def clustering(self, eval: bool = True, save: bool = True):
        if self.dr is None:
            raise ValueError("(clustering) function requires data preprocessing")
        elif self.clusters is None:
            raise ValueError("(clustering) function requires dimension_reduction")
        else:
            self.clusters.run('cnn')
            self.clusters.run('pre')
            self.clusters.run('fine')
            
            if eval:
                self.clusters.evaluate_clusters(save)
                self.clusters.evaluate_clusters_char('pre', save)
                self.clusters.evaluate_clusters_char('fine', save)
            
    
    def visualize_with_char(self):
        if self.dr is None:
            raise ValueError("(visualize_with_char) function requires data preprocessing")
        elif self.clusters is None:
            raise ValueError("(visualize_with_char) function requires dimension_reduction")
        else:
            self.plot.visualize_with_char()
        
    
    def visualize_with_char_all(self, save_gif: bool = True):
        if self.dr is None:
            raise ValueError("(visualize_with_char_all) function requires data preprocessing")
        elif self.clusters is None:
            raise ValueError("(visualize_with_char_all) function requires dimension_reduction")
        else:
            self.plot.visualize_with_char_all(save_gif)
        

    def run(self):
        try:
            self.preprocessing()
            self.dimension_reduction()
            self.clustering()
            self.visualize_with_char()
            self.visualize_with_char_all()
        except Exception as error:
            print (f'\nError: {error} \n')
            traceback.print_exc()
        