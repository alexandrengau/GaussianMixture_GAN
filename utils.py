import torch
import torchvision
import os
import json


def D_train(x, L_G, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output
    D_real_acc = ((D_output.squeeze() > 0.5) == y_real.squeeze()).float().mean().item()

    # train discriminator on fake
    # z = torch.randn(x.shape[0], 100).cuda()
    z = L_G(batch_size=x.shape[0]).cuda()
    x_fake, y_fake = G(z).cuda(), torch.zeros(x.shape[0], 1).cuda()
    
    D_output =  D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output
    
    D_fake_acc = ((D_output.squeeze() > 0.5) == y_fake.squeeze()).float().mean().item()

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward(retain_graph=True)
    D_optimizer.step()

    D_metrics = {
        "D_real_acc" : D_real_acc,
        "D_fake_acc" : D_fake_acc,
        "D_real_loss" : D_real_loss.detach().cpu().item(),
        "D_fake_loss" : D_fake_loss.detach().cpu().item()
    }
        
    return D_metrics


def G_train(x, L_G, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#


    G.zero_grad()

    z = L_G(batch_size=x.shape[0]).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward(retain_graph=True)
    G_optimizer.step()

    G_acc = ((D_output.squeeze() > 0.5) == y.squeeze()).float().mean().item()
    
    G_metrics = {
        "G_acc" : G_acc,
        "G_loss" : G_loss.detach().cpu().item()
    }
        
    return G_metrics



def save_models(L_G, G, D, folder, epoch):
    epoch_path = os.path.join(folder, f"epoch-{epoch}")
    os.makedirs(epoch_path, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(epoch_path, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(epoch_path, 'D.pth'))
    torch.save(L_G.state_dict(), os.path.join(epoch_path, 'L_G.pth'))


def load_model(L_G, G, model_dir):
    G_ckpt = torch.load(os.path.join(model_dir, 'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in G_ckpt.items()})
    L_G_ckpt = torch.load(os.path.join(model_dir,'L_G.pth'))
    L_G.load_state_dict({k.replace('module.', ''): v for k, v in L_G_ckpt.items()})
    return L_G, G, 

def load_pretrain_model(L_G, G, D, config):
    vanilla_id = config["training"]["vanilla_id"]
    epoch_checkpoint = config["training"]["epoch_checkpoint"]
    path = os.path.join("runs", vanilla_id, "pretrain/checkpoints", epoch_checkpoint)
    G_ckpt = torch.load(os.path.join(path, 'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in G_ckpt.items()})
    L_G_ckpt = torch.load(os.path.join(path,'L_G.pth'))
    L_G.load_state_dict({k.replace('module.', ''): v for k, v in L_G_ckpt.items()})
    D_ckpt = torch.load(os.path.join(path,'D.pth'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in D_ckpt.items()})
    return L_G, G, D

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def generate_sample(L_G, G, epoch, path):
    
    batch_size = 32
    n_samples = 0
    epoch_path = os.path.join(path, f"epoch-{epoch}")
    os.makedirs(epoch_path, exist_ok=True)
    while n_samples<20:  
        z = L_G(batch_size=batch_size).cuda()
        x = G(z)
        x = x.reshape(batch_size, 28, 28)
        for k in range(x.shape[0]):
            if n_samples<20:
                torchvision.utils.save_image(x[k:k+1], os.path.join(epoch_path, f'sample-{n_samples}.png'))         
                n_samples += 1

def get_run_name(config):

    law = config["latent"]["law"] 
    dim = config["latent"]["dim"]
    n_gaussian = config["latent"]["n_gaussian"]  
    learn_type = config["latent"]["learn_type"]  
    c = config["latent"]["c"]  
    sigma = config["latent"]["sigma"]
    covar_type = config["latent"]["covar_type"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]

    if law == "vanilla":
        run_id = f"law-{law}_dim-{dim}_batch_size-{batch_size}_lr-{lr}"

    elif (law == "GM") and (learn_type == "static"):
        run_id = f"law-{law}_learn_type-{learn_type}_dim-{dim}_n_gaussian-{n_gaussian}_c-{c}_sigma-{sigma}_batch_size-{batch_size}_lr-{lr}"
        
    elif (law == "GM") and (learn_type == "dynamic"):
        run_id = f"law-{law}_learn_type-{learn_type}_dim-{dim}_n_gaussian-{n_gaussian}_covar-{covar_type}_batch_size-{batch_size}_lr-{lr}"
        
    return run_id

def get_paths(config, run_id):
    trainin_type = config["training"]["type"]
    vanilla_id = config["training"]["vanilla_id"]

    if trainin_type == "finetune":
        logs_path = 'runs/' + vanilla_id + "/fine_tuning/" + run_id + "/logs"
        checkpoints_path = "runs/" + vanilla_id + "/fine_tuning/" + run_id + '/checkpoints'
        epoch_samples_path = "runs/" + vanilla_id + "/fine_tuning/" + run_id + '/samples'
        fid_temporary_path = "runs/" + vanilla_id + "/fine_tuning/" + run_id + '/fid_temporary_samples'

    elif trainin_type == "pretrain":
        logs_path = 'runs/' + run_id + "/pretrain/logs" 
        checkpoints_path = 'runs/' + run_id + "/pretrain/checkpoints"
        epoch_samples_path = "runs/" + run_id + "/pretrain/samples"
        fid_temporary_path = "runs/" + run_id + "/pretrain/fid_temporary_samples"

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(epoch_samples_path, exist_ok=True)
    os.makedirs(fid_temporary_path, exist_ok=True)

    return logs_path, checkpoints_path, epoch_samples_path, fid_temporary_path
