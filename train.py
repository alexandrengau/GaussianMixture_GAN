import torch 
import os
from tqdm import trange, tqdm
import shutil
import argparse
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3


from model import Generator, Discriminator, Latent_Generator
from utils import D_train, G_train, save_models, load_config, generate_sample, get_run_name, get_paths, load_pretrain_model

from torch.utils.tensorboard import SummaryWriter

def save_real_samples(data_loader, config):
    batch_size = config["training"]["batch_size"]
    real_images_path = config["fid"]["real_images_path"]
    os.makedirs(real_images_path, exist_ok=True)
    for batch_idx, (x, _) in enumerate(data_loader):
        if x.shape[0] != batch_size:
            image = x.reshape(x.shape[0],28,28)
        else:
            image = x.reshape(batch_size, 28, 28)
        for k in range(x.shape[0]):
            filename = os.path.join(real_images_path, f'real_image_{batch_idx * batch_size + k}.png')
            torchvision.utils.save_image(image[k:k+1], filename)


# Function to generate fake samples using the generator
def generate_fake_samples(L_G, G, config, fid_temporary_path):
    batch_size = config["training"]["batch_size"]
    num_samples = config["fid"]["num_samples"]
    L_G.eval()
    G.eval()
    n_samples = 0
    with torch.no_grad():
        while n_samples<num_samples:
            z = L_G(batch_size).cuda()
            x = G(z)
            x = x.reshape(batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<num_samples:
                    torchvision.utils.save_image(x[k:k+1], os.path.join(fid_temporary_path, f'{n_samples}.png'))
                    n_samples += 1

if __name__ == '__main__':

    config = load_config("config.json")
    
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=config["training"]["epochs"],
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=config["training"]["lr"],
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=config["training"]["batch_size"], 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--dims", type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help='Dimensionality of Inception features to use.')
    parser.add_argument("--num_workers", type=int, default=0,
                        help=('Number of processes to use for data loading. Defaults to `min(8, num_cpus)`'))

    args = parser.parse_args()
    
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    config = load_config("config.json")
    run_id = get_run_name(config)
    logs_path, checkpoints_path, epoch_samples_path, fid_temporary_path = get_paths(config, run_id)

    
    writer = SummaryWriter(logs_path)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))]) # before : transforms.Normalize(mean=(0.5), std=(0.5))

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).cuda()
    D = Discriminator(mnist_dim).cuda()
    L_G = Latent_Generator(config).cuda()
    if config["training"]["type"] == "finetune":
        L_G, G, D = load_pretrain_model(L_G, G, D, config)
    
    #G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    #D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()
    #L_G = torch.nn.DataParallel(Latent_Generator(config)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 

    # define loss
    criterion = nn.BCELoss()

    # define optimizers  
    
    G_optimizer = optim.Adam(list(G.parameters()) + list(L_G.parameters()), lr = args.lr) # if vanilla, L_G has no learnable paramters
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)


    list_G_loss = []
    list_G_acc = []
    list_D_fake_loss = []
    list_D_real_loss = []
    list_D_fake_acc = []
    list_D_real_acc = []
    list_fid = [0]
    print('Start Training :')

    # Path to save test images
    test_images_path = 'path_to_test_images_directory'

    if not os.path.exists(config["fid"]["real_images_path"]):
        # Create a DataLoader for test images
        # Save the test images
        save_real_samples(test_loader, config)
    
    n_epoch = args.epochs
    for epoch in tqdm(range(1, n_epoch+1)):  
        
        epoch_G_loss = 0
        epoch_G_acc = 0
        epoch_D_real_loss = 0
        epoch_D_fake_loss = 0
        epoch_D_real_acc = 0
        epoch_D_fake_acc = 0
             
        for batch_idx, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
            x = x.view(-1, mnist_dim)
            D_metrics = D_train(x, L_G, G, D, D_optimizer, criterion)
            G_metrics = G_train(x, L_G, G, D, G_optimizer, criterion)
            """
            if epoch == 1 :
                G_metrics = G_train(x, L_G, G, D, G_optimizer, criterion)
            if epoch % 2 == 0 :
                G_metrics = G_train(x, L_G, G, D, G_optimizer, criterion)
            """
            
            ### Update Metrics ###
            epoch_G_loss += G_metrics["G_loss"]
            epoch_G_acc += G_metrics["G_acc"]
            epoch_D_real_loss += D_metrics["D_real_loss"]
            epoch_D_fake_loss += D_metrics["D_fake_loss"]
            epoch_D_real_acc += D_metrics["D_real_acc"]
            epoch_D_fake_acc += D_metrics["D_fake_acc"]

    # CALCULATING THE FID SCORE EVERY 10 EPOCHS
        if epoch % 10 == 0:
            real_images_path = config["fid"]["real_images_path"]
            generate_fake_samples(L_G, G, config, fid_temporary_path)
            fid_value = calculate_fid_given_paths(
                [real_images_path, fid_temporary_path],
                batch_size = 128,
                device = 'cuda',
                dims = 2048
            )           
            list_fid.append(fid_value)

        list_G_loss.append(epoch_G_loss / len(train_loader))
        list_G_acc.append(epoch_G_acc / len(train_loader))
        list_D_fake_acc.append(epoch_D_fake_acc / len(train_loader))
        list_D_real_acc.append(epoch_D_real_acc / len(train_loader))
        list_D_real_loss.append(epoch_D_real_loss / len(train_loader))
        list_D_fake_loss.append(epoch_D_fake_loss / len(train_loader))

        print(
            epoch,
            "\n\t - G loss: {:.4f}".format(list_G_loss[-1]),
            "G acc: {:.2%}".format(list_G_acc[-1]), 
            "D fake loss: {:.4f}".format(list_D_fake_loss[-1]),
            "D real loss: {:.4f}".format(list_D_real_loss[-1]),
            "D fake acc: {:.2%}".format(list_D_fake_acc[-1]),
            "D real acc: {:.2%}".format(list_D_real_acc[-1]),
            "FID score: {:.4f}".format(list_fid[-1]),
        )
        
        writer.add_scalar('Generator/Loss', list_G_loss[-1], epoch)
        writer.add_scalar('Generator/Accuracy', list_G_acc[-1], epoch)
        writer.add_scalar('Discriminator/Loss/Fake', list_D_fake_loss[-1], epoch)
        writer.add_scalar('Discriminator/Loss/Real', list_D_real_loss[-1], epoch)
        writer.add_scalar('Discriminator/Accuracy/Fake', list_D_fake_acc[-1], epoch)
        writer.add_scalar('Discriminator/Accuracy/Real', list_D_real_acc[-1], epoch)
        writer.add_scalar('FID Score', list_fid[-1], epoch)
        
        generate_sample(L_G, G, epoch, epoch_samples_path)
  
        if epoch % 10 == 0:
            save_models(L_G, G, D, checkpoints_path, epoch)

    writer.close()

    # shutil.rmtree(test_images_path) # uncomment if you want to always delete 'path_to_test_images_directory/' directory after training
    
    print('Training done')

        
