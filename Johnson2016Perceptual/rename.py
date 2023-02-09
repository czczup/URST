import torch

def rename(checkpoint_model):
    checkpoint = torch.load(checkpoint_model)
    state_dict = checkpoint
    
    state_dict = {k.replace("norm", "norm.norm"): v for index, (k, v) in enumerate(state_dict.items())}
    torch.save(state_dict, checkpoint_model[:-4]+"_rename.pth")

if __name__ == '__main__':
    rename("models/cuphead_10000.pth")
    rename("models/mosaic_10000.pth")
    rename("models/starry_night_10000.pth")