def set_seed(seed: int, deterministic: bool = False):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # If there is GPU
    torch.cuda.manual_seed_all(seed)          # multi-GPU

    # 3) cuDNN flags (solo si tienes CUDA/cuDNN)
    if deterministic:
        torch.backends.cudnn.deterministic = True   # use determinist kernels
        torch.backends.cudnn.benchmark = False      
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True 
        
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def accuracy_from_logits(logits, y_true):
    import torch
    correct = torch.sum((torch.argmax(dim=1,input=logits)==y_true).int()).item()
    nelems = logits.size()[0]
    return correct/nelems

def save_checkpoint(model, optimizer, epoch, path, scheduler=None, extra: dict | None = None):
    import torch, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "extra": extra or {},
    }
    torch.save(payload, path)
    
def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location="cpu"):
    import torch
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

# It resizes the image, transforms it into a tensor and normalizes it. And [data] -> [[data]] (batch_size = 1, only the image to evaluate)
def transform_img_to_CIFAR10(path,device):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
    ])
    return transform(img).unsqueeze(0).to(device)

# It receives the logits from transform_img_to_CIFAR10
def CIFAR10_top3(logits):
    import torch
    class_dict= {
        0:"Plane",
        1:"Car",
        2:"Bird",
        3:"Cat",
        4:"Deer",
        5:"Dog",
        6:"Frog",
        7:"Horse",
        8:"Ship",
        9:"Truck"
    }
    logits = torch.softmax(logits,dim=1)
    top_vals, top_idx = logits.topk(3, largest=True, sorted=True)
    top_vals, top_idx = top_vals.flatten().tolist(), top_idx.flatten().tolist()
    msgarr = []
    for i in range(0,3):
        topindex=top_idx[i]
        msgarr.extend([str(i+1),"º ",str(top_vals[i])," | ", class_dict[topindex],"\n"])
    msg = ''.join(msgarr)
    print(msg)