
from os import path
import traceback

from .preprocessing import CNNEmbeddings, PretrainedEmbeddings, FinetunedEmbeddings, Corpus
from .dimension_reduction import DimensionalityReduction
from .clustering import Clustering
from .visualization import Visualization


class Wav2vec2Interpretation():
    def __init__(self, model_pre: str, model_fine: str, model_language: str, path_vocab: str, path_corpus: str, output_folder: str, device: str = 'cpu'):
        
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
            
        self.list_files = ['finetuned_cnn_embeddings.pkl', 'finetuned_embeddings.pkl', 'pretrained_embeddings.pkl', 'predicted_ids.pkl']
        self.list_vis = ['cnn_visualizations.pkl', 'pre_visualizations.pkl', 'fine_visualizations.pkl']
        
        self.model_pre = check_str(model_pre.strip(), 'model_pre')
        self.model_fine = check_str(model_fine.strip(), 'model_pre')
        self.model_language = check_str(model_language.strip(), 'model_language')
        
        self.path_vocab = check_str(path_vocab.strip(), 'path_vocab')
        self.path_corpus = check_str(path_corpus.strip(), 'path_corpus')
        self.output_folder = check_str(output_folder.strip(), 'output_folder')
        self.device = check_str(device.strip(), 'output_folder')
        
        if not path.isfile(self.path_vocab):
            raise ValueError(f"(path_vocab) not found")
        if self.model_language is not None and not path.isfile(self.model_language):
            raise ValueError(f"(model_language) not found")
        
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
           
        self.dr = None
        self.clusters = None
        self.plot = None
        
        
    def preprocessing(self):    
        print (f"\n---> Starting preprocessing on device='{self.device}'\n")   
        
        self.corpus = Corpus(self.output_folder, self.path_corpus, self.model_language, self.device)
        self.cnn_embed = CNNEmbeddings(self.output_folder, self.path_corpus, self.device)
        self.pre_embed = PretrainedEmbeddings(self.output_folder, self.path_corpus, self.device)
        self.fine_embed = FinetunedEmbeddings(self.output_folder, self.path_corpus, self.device)
        
        print (f"\t1- Corpus (predicts_ids)...") 
        self.corpus.extract(self.model_fine)
        print (f"\n\t2- CNN embeddings...") 
        self.cnn_embed.extract(self.model_fine)
        print (f"\n\t3- Pretrained transformers layers...") 
        self.pre_embed.extract(self.model_pre)
        print (f"\n\t4- Finetuned transformers layers...") 
        self.fine_embed.extract(self.model_fine)    
        
    
    def check_files(self, vis: bool = False) -> bool:
        output_folder = self.output_folder + '/wav2vec2_interpretation'
        
        if not path.exists(output_folder + ''):
            raise ValueError(f"({output_folder}) not found, function requires data preprocessing ")
        
        for f in self.list_files:
            path_file = path.join(output_folder, 'data/' + f)
            if not path.isfile(path_file):
                raise ValueError(f"({path_file}) not found, function requires data preprocessing")
        
        if vis:
            for f in self.list_vis:
                path_file = path.join(output_folder, 'data/' + f)
                if not path.isfile(path_file):
                    raise ValueError(f"({path_file}) not found, function requires dimension_reduction")
            
        return True
     
        
    def dimension_reduction(self, type: str = 'all'):
        # require preprocessing     
        if self.check_files():
            print (f"\n---> Starting dimensionality reduction")  
            self.dr = DimensionalityReduction(self.output_folder)
            if type == 'pca':
                print (f"\n\tPCA...") 
                self.dr.extract_pca('cnn')
                self.dr.extract_pca('pre')
                self.dr.extract_pca('fine')
            elif type == 'tsne':
                print (f"\n\tt-SNE...") 
                self.dr.extract_tsne('cnn')
                self.dr.extract_tsne('pre')
                self.dr.extract_tsne('fine')
            elif type == 'umap':
                print (f"\n\tUMAP...")
                self.dr.extract_umap('cnn')
                self.dr.extract_umap('pre')
                self.dr.extract_umap('fine')
            elif type == 'all':
                print (f"\n\tCNN embeddings (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('cnn')
                print (f"\n\tPretrained transformers layers (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('pre')
                print (f"\n\tFinetuned transformers layers (PCA, t-SNE and UMAP)...")
                self.dr.extract_all('fine')
            else:
                raise ValueError("(type) parameter is invalid require: 'cnn', 'pre', 'fine' or 'all'.")
            
            print (f"\nPlots saved on '{self.output_folder}/wav2vec2_interpretation/images'")
            
            
    def clustering(self, eval: bool = True, save: bool = True) -> dict:
        print (f"\n---> Starting clustering")
        self.clusters = Clustering(self.output_folder, self.path_vocab)
        results = dict()
        
        if self.check_files(vis=True):
            self.clusters.run('cnn')
            self.clusters.run('pre')
            self.clusters.run('fine')
            
            if eval:
                results['metrics'] = self.clusters.evaluate_clusters(save)
                results['pretrained'] = self.clusters.evaluate_clusters_char('pre', save)
                results['finetuned'] = self.clusters.evaluate_clusters_char('fine', save)
                if save:
                    print (f"\nResults saved on '{self.output_folder}/wav2vec2_interpretation/data'")
        
            print (f"\nPlots saved on '{self.output_folder}/wav2vec2_interpretation/images'")
        return results
            
    
    def visualize_with_char(self):
        print (f"\n---> Plot visualization (char)")
        self.plot = Visualization(self.output_folder, self.path_vocab)
        
        if self.check_files(vis=True):
            self.plot.visualize_with_char('cnn')
            self.plot.visualize_with_char('pre')
            self.plot.visualize_with_char('fine')
        
            print (f"\nPlots saved on '{self.output_folder}/wav2vec2_interpretation/images'")
        
        
    def visualize_with_char_all(self, save_gif: bool = False):
        print (f"\n---> Plot visualization char (all)")
        self.plot = Visualization(self.output_folder, self.path_vocab)
        
        if self.check_files(vis=True):
            self.plot.visualize_with_char_all(save_gif)
        
            print (f"\nPlots saved on '{self.output_folder}/wav2vec2_interpretation/images'")


    def run(self):
        try:
            self.preprocessing()
            self.dimension_reduction()
            self.clustering(eval=True)
            self.visualize_with_char()
            self.visualize_with_char_all()
        except Exception as error:
            print (f'\nError: {error} \n')
            traceback.print_exc()
        