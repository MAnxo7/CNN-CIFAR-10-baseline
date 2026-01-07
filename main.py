import argparse
import torch
from src import utils, models, train, data

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50, help="The number of epochs that are used for model training.")
    p.add_argument("--batch-size", type=int, default=128, help="The batch-size that is used for model training")
    p.add_argument("--lr", type=float, default=0.01,help="Learning rate for training")
    p.add_argument("--eval-only", action="store_true",help="Evaluation mode: You need a checkpoint path with --ckpt-path and optionally an --image-path for inference")
    p.add_argument("--device", type=str, default=utils.get_device(),help="The device where the model is going to work (CUDA/CPU)")
    p.add_argument("--ckpt-path", type=str, default=None, help="Load the model from a checkpoint, if not --eval-only the train will continue from it")
    p.add_argument("--image-path", type=str, default=None, help="Inferences from an image. If not --eval-only, first the model will train and then it will inference")
    p.add_argument("--disable-sched", action="store_true", help="Disables the multistep scheduler (milestones=[30, 40], gamma=0.3)")
    p.add_argument("--arch", type=str, default="exp21", choices=["exp0","exp15","exp21"],help="Select an architecture. Options: exp0 ,exp15, exp21. Default: exp21 ")
    return p.parse_args()

def main():
    args = parse_args()

    utils.set_seed(0, deterministic=True)
    device = args.device

    train_loader, test_loader, image_sample = data.get_cifar10_loaders(args.batch_size, show_only_transf=False)
    # Load the correspondent model
    class_name = "BasicCNN_" + args.arch
    ModelArch = getattr(models, class_name)      
    model = ModelArch(image=image_sample, device=device)
    # Other model variables
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = None if args.disable_sched else torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 40], gamma=0.3)
    epochs = args.epochs

    loss_fn = torch.nn.CrossEntropyLoss()

    # Load the checkpoint and get the remaining epochs if you are continuing a training
    if args.ckpt_path:
        try:
            ckpt = utils.load_checkpoint(args.ckpt_path,model,opt,scheduler=scheduler)
            epochs = max(0, epochs - ckpt["epoch"])
            
        except RuntimeError:
            raise ValueError("The current architecture isn't compatible with this checkpoint")

    # Eval or train?
    if args.eval_only:    
        if args.ckpt_path is None:
            raise ValueError("You should specify --ckpt-path when using --eval-only")
        val_metrics = train.evaluate(model,test_loader,loss_fn,device) 
        print(val_metrics)

    else:
        train.fit(model,device,train_loader,test_loader,opt,loss_fn,epochs,early_stopping=5, scheduler=scheduler)

    # Returns the top3 prediction of the model if you have attached an image.
    if args.image_path:
        model.eval()
        with torch.no_grad():
            x = utils.transform_img_to_CIFAR10(path=args.image_path,device=device)
            logits = model(x)
            utils.CIFAR10_top3(logits)


    
    del train_loader, test_loader

if __name__ == "__main__":
    main()

