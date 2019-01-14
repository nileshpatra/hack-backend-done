import torch
import torchvision
import pickle
from torch import optim 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import apple_call

file = open('pred.txt' , 'r')

print('FILLLLLEEEEEEEEEEEE DATAAAAAAAAA : ')
s = file.read()
print(s)
test_img_transforms = transforms.Compose([transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
#train_data = datasets.ImageFolder('/home/nilesh/Desktop/MY FILES/hackathon-dataset/raw/apple_data/training', transform=train_transforms)
test_data = datasets.ImageFolder(s, transform=test_img_transforms)

testloader = torch.utils.data.DataLoader(test_data, batch_size=1 , shuffle = True)

_ , (images , _) = next(enumerate(testloader))





print('===========================================================================')
nn = neural_net()
filename = 'finalized_model1.sav'
loaded_model = pickle.load(open(filename ,'rb'))

pred = torch.exp(loaded_model(images)).argmax().detch().numpy()
print(pred)
pred_path = file('predicted.txt' , 'w+')
file.write(pred)

work()