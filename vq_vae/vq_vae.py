
import torch.nn as nn
from encoder import Encoder
from quantizer import VectorQuantizer
from decoder import Decoder


class VQVAE(nn.Module):
    def __init__(
            self, h_dim, res_h_dim, n_res_layers,
            n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity

    def encode_to_indices(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, _, _, _, indices = self.vector_quantization(z_e)

        # 获取 batch_size, height, width
        batch_size, _, height, width = z_e.shape
        
        # indices 可能有多余的维度，先 squeeze
        indices = indices.squeeze()  # shape: [N*H*W]
        
        # reshape 成 [N, H, W]
        indices = indices.view(batch_size, height, width)

        return indices
