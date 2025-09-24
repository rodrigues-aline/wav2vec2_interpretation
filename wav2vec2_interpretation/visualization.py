from abc import ABC

from distinctipy import distinctipy
from cairosvg import svg2png
from json import load
from glob import glob
from PIL import Image

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


class Visualization(ABC):
    def __init__(self, output_folder: str, path_vocab: str):
        self.output_folder = output_folder + '/wav2vec2_interpretation'
        self.path_vocab = path_vocab
        
        self.vocab = load(open(self.path_vocab))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.tokens_ignore = ["<pad>", "|", "<unk>", "-", "<s>", "</s>", "[UNK]", "[PAD]", " "]
        self.indices_ignore = [self.vocab[t] for t in self.tokens_ignore if t in self.vocab]
        
        cnn_pca,  cnn_tsne,  cnn_umap = pkl.load(open(f'{self.output_folder}/data/cnn_visualizations.pkl', 'rb'))
        pre_pca,  pre_tsne,  pre_umap = pkl.load(open(f'{self.output_folder}/data/pre_visualizations.pkl', 'rb'))
        fine_pca, fine_tsne, fine_umap = pkl.load(open(f'{self.output_folder}/data/fine_visualizations.pkl', 'rb'))
        
        self.data_comp = dict(pre  = dict(pca=pre_pca,  tsne=pre_tsne,  umap=pre_umap),
                              fine = dict(pca=fine_pca, tsne=fine_tsne, umap=fine_umap),
                              cnn  = dict(pca=cnn_pca,  tsne=cnn_tsne,  umap=cnn_umap))
        
        self.lab_reco = pkl.load(open(f'{self.output_folder}/data/predicted_ids.pkl', 'rb'))
        
        self.lab = []

        for l in self.lab_reco:
            self.lab = np.concatenate((self.lab, self.lab_reco[l]))
        self.lab = np.array(self.lab)
        
        self.legend = list(self.inv_vocab.values())
        self.cmaps = distinctipy.get_colors(len(self.legend))
        
        
    def visualize_with_char(self, type_embedding: str):
        for i in range(len(self.legend)):
            if i in self.indices_ignore:
                continue
            idx = np.where(self.lab==i)
            plt.scatter(self.data_comp[type_embedding]['tsne'][idx, 0], self.data_comp[type_embedding]['tsne'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_tsne_char.svg", bbox_inches='tight')
        plt.close('all')
        plt.figure().clear()
        
        for i in range(len(self.legend)):
            if i in self.indices_ignore:
                continue
            idx = np.where(self.lab==i)
            plt.scatter(self.data_comp[type_embedding]['umap'][idx, 0], self.data_comp[type_embedding]['umap'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 3, markerscale=16)
        plt.savefig(f"{self.output_folder}/images/{type_embedding}_umap_char.svg", bbox_inches = 'tight')
        plt.close('all')
        plt.figure().clear()
        
    
    def save_svg_to_gif(self, mode: str):
        images_png = []

        images = glob(f'{self.output_folder}/images/{mode}_*.svg')
        images.sort()

        for image_path in images:
            svg2png(url=image_path, write_to=image_path.replace('.svg', '.png'))
            images_png.append(image_path.replace('.svg', '.png'))

        image_list = [Image.open(img) for img in images_png]

        image_list[0].save(f'{self.output_folder}/images/{mode}_char.gif', save_all=True, append_images=image_list[1:], duration=1000, loop=0)
        
        
    def visualize_with_char_all(self, save_gif: bool = True):
        for i in range(len(self.legend)):
            if i in self.indices_ignore:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            idx = np.where(self.lab == i)

            axs[0].scatter(self.data_comp['cnn']['umap'][idx, 0], self.data_comp['cnn']['umap'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])
            axs[1].scatter(self.data_comp['pre']['umap'][idx, 0], self.data_comp['pre']['umap'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])
            axs[2].scatter(self.data_comp['fine']['umap'][idx, 0], self.data_comp['fine']['umap'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])

            axs[0].legend(loc='upper right', markerscale=16)
            axs[1].legend(loc='upper right', markerscale=16)
            axs[2].legend(loc='upper right', markerscale=16)

            axs[0].set_title(f'CNN')
            axs[1].set_title(f'Pretrained')
            axs[2].set_title(f'Finetuned')

            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/images/umap_{self.inv_vocab[i]}.svg", bbox_inches='tight')
            plt.close()

        for i in range(len(self.legend)):
            if i in self.indices_ignore:
                continue
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            idx = np.where(self.lab == i)

            axs[0].scatter(self.data_comp['cnn']['tsne'][idx, 0], self.data_comp['cnn']['tsne'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])
            axs[1].scatter(self.data_comp['pre']['tsne'][idx, 0], self.data_comp['pre']['tsne'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])
            axs[2].scatter(self.data_comp['fine']['tsne'][idx, 0], self.data_comp['fine']['tsne'][idx, 1], s=0.01, color=self.cmaps[i], label=self.inv_vocab[i])

            axs[0].legend(loc='upper right', markerscale=16)
            axs[1].legend(loc='upper right', markerscale=16)
            axs[2].legend(loc='upper right', markerscale=16)

            axs[0].set_title(f'CNN')
            axs[1].set_title(f'Pretrained')
            axs[2].set_title(f'Finetuned')

            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/images/tsne_{self.inv_vocab[i]}.svg", bbox_inches='tight')
            plt.close()
            
            if save_gif:
                self.save_svg_to_gif('umap')
                