from abc import ABC

from sklearn.cluster import KMeans
from coclust.clustering.spherical_kmeans import SphericalKmeans
from distinctipy import distinctipy
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, homogeneity_score, completeness_score, silhouette_score

from json import load, dump

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


class Clustering(ABC):
    def __init__(self, output_folder: str, path_vocab: str, model_language: bool = False):
        self.output_folder = output_folder + '/wav2vec2_interpretation'
        self.path_vocab = path_vocab
        self.model_language = model_language
        
        self.vocab = load(open(self.path_vocab))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        pre  = pkl.load(open(f"{self.output_folder}/data/pretrained_embeddings.pkl", "rb"))
        fine = pkl.load(open(f"{self.output_folder}/data/finetuned_embeddings.pkl", "rb"))
        cnn  = pkl.load(open(f"{self.output_folder}/data/finetuned_cnn_embeddings.pkl", "rb"))
        
        self.data = dict(pre  = np.concatenate(tuple(pre.values()), axis=1)[0],
                         fine = np.concatenate(tuple(fine.values()), axis=1)[0],
                         cnn  = np.concatenate(tuple(cnn.values()), axis=1)[0])
        
        cnn_pca,  cnn_tsne,  cnn_umap = pkl.load(open(f'{self.output_folder}/data/cnn_visualizations.pkl', 'rb'))
        pre_pca,  pre_tsne,  pre_umap = pkl.load(open(f'{self.output_folder}/data/pre_visualizations.pkl', 'rb'))
        fine_pca, fine_tsne, fine_umap = pkl.load(open(f'{self.output_folder}/data/fine_visualizations.pkl', 'rb'))
        
        self.data_comp = dict(pre  = dict(pca=pre_pca,  tsne=pre_tsne,  umap=pre_umap),
                              fine = dict(pca=fine_pca, tsne=fine_tsne, umap=fine_umap),
                              cnn  = dict(pca=cnn_pca,  tsne=cnn_tsne,  umap=cnn_umap))
        
        self.lab =  np.concatenate(tuple(pkl.load(open(f'{self.output_folder}/data/predicted_ids.pkl', 'rb')).values()))
        
        if self.model_language:
            self.lab_lm =  np.concatenate(tuple(pkl.load(open(f'{self.output_folder}/data/predicted_ids_lm.pkl', 'rb')).values()))
        
        #self.reco_lab = pkl.load(open(f'{self.output_folder}/data/predicted_ids.pkl', 'rb'))
        
    
    def convert_types(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
    
    def evaluate_clusters(self, save: bool = True, model_language: bool = False) -> dict:
        labs_cnn, _,  _, _              = pkl.load(open(f"{self.output_folder}/data/cnn_kmeans_clusters.pkl","rb"))
        labs_pre, _,  labs_pre_umap, _  = pkl.load(open(f"{self.output_folder}/data/pre_kmeans_clusters.pkl","rb"))
        labs_fine, _, labs_fine_umap, _ = pkl.load(open(f"{self.output_folder}/data/fine_kmeans_clusters.pkl","rb"))
        
        
        lab = self.lab_lm if model_language else self.lab

        metrics_asr = dict()
        for f in [adjusted_mutual_info_score, mutual_info_score, homogeneity_score, completeness_score]:
            m = str(f).split(' ')[1]
            metrics_asr[m] = dict()

            metrics_asr[m]["CNN vs Predictions"] = f(lab, labs_cnn)
            metrics_asr[m]["CNN vs Pre"]         = f(labs_pre, labs_cnn)
            metrics_asr[m]["CNN vs Fine"]        = f(labs_fine, labs_cnn)

            metrics_asr[m]["Predictions vs Pre UMAP"] = f(lab, labs_pre_umap)
            metrics_asr[m]["Predictions vs Pre"]      = f(lab, labs_pre)
            metrics_asr[m]["Pre vs Pre UMAP"]         = f(labs_pre, labs_pre_umap)

            metrics_asr[m]["Predictions vs Fine UMAP"] = f(lab, labs_fine_umap)
            metrics_asr[m]["Predictions vs Fine"]      = f(lab, labs_fine)
            metrics_asr[m]["Fine vs Fine UMAP"]        = f(labs_fine, labs_fine_umap)

            metrics_asr[m]["Fine vs Pre"]      = f(labs_fine, labs_pre)
            metrics_asr[m]["Fine vs Pre UMAP"] = f(labs_fine_umap, labs_pre_umap)

        m = 'silhouette_scores'
        metrics_asr[m] = dict()

        metrics_asr[m]['pretrained UMAP'] = silhouette_score(self.data['pre'], labs_pre_umap, metric='cosine')
        metrics_asr[m]['pretrained']      = silhouette_score(self.data['pre'], labs_pre, metric='cosine')
        metrics_asr[m]['finetuned']       = silhouette_score(self.data['fine'], labs_fine_umap, metric='cosine')
        metrics_asr[m]['finetuned']       = silhouette_score(self.data['fine'], labs_fine, metric='cosine')
        
        if save:
            with open(f'{self.output_folder}/data/eval_clusters_metrics{'_ml' if model_language else ''}.json', 'w') as f:
                dump(metrics_asr, f, indent=4, ensure_ascii=False, default=self.convert_types)
        
        return metrics_asr
        
    
    def evaluate_clusters_char(self, type_embedding: str, save: bool = True, model_language: bool = False) -> dict:
        labs, _, _, _ = pkl.load(open(f"{self.output_folder}/data/{type_embedding}_kmeans_clusters.pkl","rb"))        
        discovered_map = {}
        clusters_char ={}
        
        lab = self.lab_lm if model_language else self.lab

        for idx in range(len(self.inv_vocab)):
            # Ensure labs_pre is a numpy array and lab==idx results in a boolean array
            data = np.array(labs)[np.array(lab==idx)]
            # Use np.arange to create bins for cluster labels
            distr,_ = np.histogram(data, bins=np.arange(np.max(labs) + 2)) # +2 to include all possible cluster labels
            max_category = np.argmax(distr)
            if str(max_category) in discovered_map.keys():
                discovered_map[str(max_category)].append(self.inv_vocab[idx])
            else:
                discovered_map[str(max_category)] = [self.inv_vocab[idx]]
            percent = distr[max_category]/np.sum(distr)
            clusters_char[self.inv_vocab[idx]] = {'max_category': float(max_category), 'percent': str(percent)+'%'}

        mapping = {}
        for idx in range(np.max(labs)+1):
            data = lab[labs==idx]
            distr,_ = np.histogram(data, list(np.arange(np.max(labs))))
            max_category = np.argmax(distr[1:])
            mapping[idx] = max_category
            
        if save:
            with open(f'{self.output_folder}/data/eval_clusters_discovered_map_{type_embedding}{'_ml' if model_language else ''}.json', 'w') as f:
                dump(discovered_map, f, indent=4, ensure_ascii=False)

            with open(f'{self.output_folder}/data/eval_clusters_char_{type_embedding}{'_ml' if model_language else ''}.json', 'w') as f:
                dump(clusters_char, f, indent=4, ensure_ascii=False)
                           
        return dict(discovered_map=discovered_map, mapping=mapping)
    
    
    def run(self, type_embedding: str):
        if type_embedding == 'cnn':
            self.clustering_cnn()
        else:
            self.clustering_pre_fine(type_embedding)
            
    
    def clustering_pre_fine(self, type_embedding: str):
        clustering = SphericalKmeans(n_clusters=37, random_state=0)
        clustering.fit(self.data[type_embedding])
        labs = clustering.labels_

        clustering_pca = SphericalKmeans(n_clusters=37, random_state=0)
        clustering_pca.fit(self.data_comp[type_embedding]['pca'])
        labs_pca = clustering_pca.labels_

        clustering_umap = SphericalKmeans(n_clusters=37, random_state=0)
        clustering_umap.fit(self.data_comp[type_embedding]['umap'])
        labs_umap = clustering_umap.labels_

        clustering_tsne = SphericalKmeans(n_clusters=37, random_state=0)
        clustering_tsne.fit(self.data_comp[type_embedding]['tsne'])
        labs_tsne = clustering_tsne.labels_

        cmaps = distinctipy.get_colors(np.max(labs) + 1)
        unique_labels = np.unique(labs)

        for i in unique_labels:
            idx = np.where(labs==i)
            plt.scatter(self.data_comp[type_embedding]['tsne'][idx[0], 0], self.data_comp[type_embedding]['tsne'][idx[0], 1], s=0.01, color =cmaps[i], label = "cluster_"+str(i))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, markerscale=16)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_tsne_cluster_kmeans.eps", bbox_inches='tight')
        plt.close('all')
        plt.figure().clear()
        
        for i in unique_labels:
            idx = np.where(labs==i)
            plt.scatter(self.data_comp[type_embedding]['umap'][idx[0], 0], self.data_comp[type_embedding]['umap'][idx[0], 1], s=0.01, color =cmaps[i], label = "cluster_"+str(i))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3, markerscale=16)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_umap_cluster_kmeans.eps", bbox_inches='tight')
        plt.close('all')
        plt.figure().clear()

        pkl.dump([labs, labs_pca, labs_umap, labs_tsne], open(f"{self.output_folder}/data/{type_embedding}_kmeans_clusters.pkl", "wb"))
        
    
    def clustering_cnn(self):
        clustering_cnn = KMeans(n_clusters=37, random_state=0).fit(self.data['cnn'])
        labs_cnn = clustering_cnn.labels_

        clustering_cnn_vis = KMeans(n_clusters=37, random_state=0).fit(self.data_comp['cnn']['umap'])
        labs_cnn_umap = clustering_cnn_vis.labels_

        clustering_cnn_vis_pca = KMeans(n_clusters=37, random_state=0).fit(self.data_comp['cnn']['pca'])
        labs_cnn_pca = clustering_cnn_vis_pca.labels_

        clustering_cnn_vis_tsne = KMeans(n_clusters=37, random_state=0).fit(self.data_comp['cnn']['tsne'])
        labs_cnn_tsne = clustering_cnn_vis_tsne.labels_

        pkl.dump([labs_cnn, labs_cnn_pca, labs_cnn_umap, labs_cnn_tsne], open(f"{self.output_folder}/data/cnn_kmeans_clusters.pkl", "wb"))
    