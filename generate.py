import torch 
import torchvision
import os
import argparse

from tqdm import tqdm
from model import Generator, Latent_Generator
from utils import load_model, load_config

config = load_config("config.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=config["training"]["batch_size"],
                      help="The batch size to use for training.")
    args = parser.parse_args()


    model_dir = "checkpoints"
    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    L_G = Latent_Generator(config).cuda()
    G = Generator(g_output_dim = mnist_dim).cuda()
    L_G, G = load_model(L_G, G, model_dir)
    #L_G = torch.nn.DataParallel(L_G).cuda()
    #G = torch.nn.DataParallel(G).cuda()
    L_G.eval()
    G.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    total_samples = 10000
    
    with torch.no_grad():
        with tqdm(total=total_samples, desc="Generating samples") as pbar:
            while n_samples<10000:
                # z = torch.randn(args.batch_size, 100).cuda()
                z = L_G(batch_size=args.batch_size).cuda()
                x = G(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                        n_samples += 1
                        pbar.update(1)


    
