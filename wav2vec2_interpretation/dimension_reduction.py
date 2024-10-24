from abc import ABC

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import umap
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


class DimensionalityReduction(ABC):
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.seed = 1024
        
        pre  = pkl.load(open(f"{self.output_folder}/data/pretrained_embeddings.pkl", "rb"))
        fine = pkl.load(open(f"{self.output_folder}/data/finetuned_embeddings.pkl", "rb"))
        cnn  = pkl.load(open(f"{self.output_folder}/data/finetuned_cnn_embeddings.pkl", "rb"))
        
        self.data = dict(pre  = np.concatenate(tuple(pre.values()), axis=1)[0],
                         fine = np.concatenate(tuple(fine.values()), axis=1)[0],
                         cnn  = np.concatenate(tuple(cnn.values()), axis=1)[0])
        
    
    def extract_pca(self, type_embedding: str):
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.data[type_embedding])
        plt.scatter(data_pca[:,0], data_pca[:,1], s=0.01)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_pca.svg")
        plt.close('all')
        plt.figure().clear()
        
        return data_pca
        
        
    def extract_tsne(self, type_embedding: str): 
        t_sne = TSNE(n_components=2,random_state=self.seed)
        data_tsne = t_sne.fit_transform(self.data[type_embedding])
        plt.scatter(data_tsne[:,0], data_tsne[:,1], s=0.01)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_tsne.svg")
        plt.close('all')
        plt.figure().clear()
        
        return data_tsne
    
    
    def extract_umap(self, type_embedding: str): 
        umap_emb =umap.UMAP(n_components = 2, metric='cosine',random_state=self.seed)
        data_umap = umap_emb.fit_transform(self.data[type_embedding])
        plt.scatter(data_umap[:,0], data_umap[:,1], s=0.01)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_umap.svg")
        plt.close('all')
        plt.figure().clear()
        
        return data_umap
    
    
    def extract_all(self, type_embedding: str):
        data_pca = self.extract_pca(type_embedding)
        data_tsne = self.extract_tsne(type_embedding)
        data_umap = self.extract_umap(type_embedding)
        
        pkl.dump([data_pca, data_tsne, data_umap], open(f'{self.output_folder}/data/{type_embedding}_visualizations.pkl', 'wb'))
    