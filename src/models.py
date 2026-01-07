import torch
class BasicCNN_exp21(torch.nn.Module):
    def __init__(self,image,device):
        
        super().__init__()
        output = 10
        activation = torch.nn.ReLU
        
        image_H = image.shape[1]
        image_W = image.shape[2]
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding=1), #Returns 16 patron's representations of 32*32 size
            torch.nn.BatchNorm2d(num_features=16),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            activation(),
            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(in_channels=64,out_channels=96,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=96),
            activation(),
        ).to(device)
        
        dummy = torch.randn(1, 3, image_H, image_W).to(device)
        dummy_out = self.features(dummy)        
        flat_size = dummy_out.numel()           
        self.flat_size = flat_size
        
 
        nlayers = 4
        layers = [torch.nn.Flatten()]
        neurons = flat_size

        for i in range(0,nlayers):
            layers.append(torch.nn.Linear(neurons,int(neurons/2)))
            layers.append(activation())
            layers.append(torch.nn.Dropout(0.1*(nlayers-i)))
            neurons = int(neurons/2)
        layers.append(torch.nn.Linear(neurons,output))
        print("flat_size =",flat_size,"\nlast_neurons = ",neurons)
        
        self.clasiffier = torch.nn.Sequential(*layers).to(device)

        
        
    def forward(self,x):
        x = self.features(x)
        x = self.clasiffier(x)
        return x
        
class BasicCNN_exp15(torch.nn.Module):
    def __init__(self,image,device):
        
        super().__init__()
        output = 10
        activation = torch.nn.ReLU
        
        image_H = image.shape[1]
        image_W = image.shape[2]
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding=1), #Returns 16 patron's representations of 32*32 size
            torch.nn.BatchNorm2d(num_features=16),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            activation(),
            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),     
        ).to(device)
        
        dummy = torch.randn(1, 3, image_H, image_W).to(device)
        dummy_out = self.features(dummy)        
        flat_size = dummy_out.numel()           
        self.flat_size = flat_size
        
 
        nlayers = 4
        layers = [torch.nn.Flatten()]
        neurons = flat_size

        for i in range(0,nlayers):
            layers.append(torch.nn.Linear(neurons,int(neurons/2)))
            layers.append(activation())
            layers.append(torch.nn.Dropout(0.1*(nlayers-i)))
            neurons = int(neurons/2)
        layers.append(torch.nn.Linear(neurons,output))
        print("flat_size =",flat_size,"\nlast_neurons = ",neurons)
        
        self.clasiffier = torch.nn.Sequential(*layers).to(device)

        
        
    def forward(self,x):
        x = self.features(x)
        x = self.clasiffier(x)
        return x
        
class BasicCNN_exp0(torch.nn.Module):
    def __init__(self,image,device):
        
        super().__init__()
        output = 10
        activation = torch.nn.ReLU
        
        image_H = image.shape[1]
        image_W = image.shape[2]
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding=1), #Returns 16 patron's representations of 32*32 size
            torch.nn.BatchNorm2d(num_features=16),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            activation(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
    
        ).to(device)
        
        dummy = torch.randn(1, 3, image_H, image_W).to(device)
        dummy_out = self.features(dummy)        
        flat_size = dummy_out.numel()           
        self.flat_size = flat_size
        
 
        nlayers = 2
        layers = [torch.nn.Flatten()]
        neurons = flat_size

        for i in range(0,nlayers):
            layers.append(torch.nn.Linear(neurons,int(neurons/2)))
            layers.append(activation())
            neurons = int(neurons/2)
        layers.append(torch.nn.Linear(neurons,output))
        print("flat_size =",flat_size,"\nlast_neurons = ",neurons)
        
        self.clasiffier = torch.nn.Sequential(*layers).to(device)
     
        
    def forward(self,x):
        x = self.features(x)
        x = self.clasiffier(x)
        return x