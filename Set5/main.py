import torch
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import os
import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import PIL

model = resnet18(weights= ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, 5)




class LoadDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = cv2.imread(image_path)
        # convert to standard RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert to numpy array
        image = np.array(image)
        
        # Apply transformations 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)




# Create dataset
traindataset = LoadDataset('data/train')
testdataset = LoadDataset('data/val')

traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
testdataloader =  DataLoader(testdataset, batch_size=1, shuffle=False)


def extract_features(image, model):
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()


def save_features(model, folder_path='./Images', train_name = 'features_train', test_name = 'features_test' ):
    data_folder_list = os.listdir(folder_path)

    features_list_test = []
    feature_name_test = []

    feature_name_train = []
    features_list_train = []
    

    preprocessing = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                        transforms.ToTensor(),transforms.Normalize(
                                                                mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

    
    for folder_name in data_folder_list:
        if '.DS_Store' in folder_name:
            continue
        img_list = os.listdir(os.path.join(folder_path, folder_name))

        for img_name in img_list:

            if '.DS_Store' in img_name:
                continue

            # img = cv2.imread(os.path.join(folder_path, folder_name, img_name))
            # print(img_name)
            img = Image.open(os.path.join(folder_path, folder_name, img_name)).convert('RGB')

            # img_tensor = torch.tensor(img).permute(2,0,1).to(torch.float)
            # img_tensor = img_tensor.unsqueeze(0)
            img_tensor = preprocessing(img).unsqueeze(0)
            # print(img_tensor.shape)
            features = model(img_tensor)
            # print(features.detach())
            # print(features.shape)

            if '_test' in folder_name:
                feature_name_test.append(folder_name)
                features_list_test.append(features.detach().numpy()[0,:,0,0])
            
            if '_train' in folder_name:
                feature_name_train.append(folder_name)
                features_list_train.append(features.detach().numpy()[0,:,0,0])
            
            # print(features[0,:,0,0].shape)

        # print(features_list_train)
        dict_train = {'folder_name': feature_name_train, 'features': features_list_train}
        dict_test = {'folder_name': feature_name_test, 'features': features_list_test}

        df_test = pd.DataFrame(dict_test)
        df_train = pd.DataFrame(dict_train)

        df_train.to_csv(train_name +'.csv')
        df_test.to_csv(test_name + '.csv')

    print(np.array(features_list_train).shape)
    with open(train_name + '.npy', 'wb') as f:
        np.save(f, np.array(features_list_train))
    
    with open(test_name + '.npy', 'wb') as f:
        np.save(f, np.array(features_list_test))



    return features_list_train, features_list_test
            

def problem1(features_train, features_train_name, features_test, features_test_name):

    # category_train = []
    # category_test = []

    # for i in range(len(features_train_name)):
    #     if 'bird' in features_train_name[i]:
    #         category_train.append(0)
    #     elif 'butterfly' in features_train_name[i]:
    #         category_train.append(1)
    #     elif 'dog' in features_train_name[i]:
    #         category_train.append(2)
    #     elif 'Fish' in features_train_name[i]:
    #         category_train.append(3)
    #     elif 'mountains' in features_train_name[i]:
    #         category_train.append(4)

    # for i in range(len(features_test_name)):
    #     if 'bird' in features_test_name[i]:
    #         category_test.append(0)
    #     elif 'butterfly' in features_test_name[i]:
    #         category_test.append(1)
    #     elif 'dog' in features_test_name[i]:
    #         category_test.append(2)
    #     elif 'Fish' in features_test_name[i]:
    #         category_test.append(3)
    #     elif 'mountains' in features_test_name[i]:
    #         category_test.append(4)


    neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    neigh.fit(features_train, features_train_name)

    y = neigh.predict(features_test)

    print(np.sum(features_test_name == y), len(y))
    # print(neigh.get_params())
    # print(neigh.score(features_test, category_test[:]))



# layers = list(model.children())[:-1]
# feature_extractor = nn.Sequential(*layers)


# features_train, features_test = save_features(feature_extractor)
# exit()


############################# Problem 1 #################################



# layers = list(model.children())[:-1]
# feature_extractor = nn.Sequential(*layers)


layers = list(model.children())[:-1]



feature_extractor = nn.Sequential(*layers)


# features_train_csv = pd.read_csv('features_train.csv').values
# features_test_csv = pd.read_csv('features_test.csv').values


# features_test = np.load('features_test.npy')
# features_train = np.load('features_train.npy')

# print(features_train.shape, features_test.shape)

train_features = []
train_labels = []

for batch_idx, (image, label) in enumerate(traindataloader):
    features = extract_features(image, feature_extractor)
    train_features.append(features)
    label = label.numpy()
    train_labels.append(label)

test_features = []
test_labels = []

for batch_idx, (image, label) in enumerate(testdataloader):
    features = extract_features(image, feature_extractor)
    test_features.append(features)
    label = label.numpy()
    test_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(-1)

test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1)


print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)

problem1(train_features, train_labels, test_features, test_labels)

# exit()

######################### Problem 2 ###########################
# problem2(model)
lr = 1e-4

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



for i, param in enumerate(model.parameters()):
    param.requires_grad = False

# for i, param in enumerate(model.fc.parameters()):
#     param.requires_grad = True

# print(list(model.children())[-3])

for i, param in enumerate(list(model.children())[-3].parameters()):
    param.requires_grad = True

#preprocessing
transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                        transforms.ToTensor(),transforms.Normalize(
                                                                mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])

train_dataset = ImageFolder(root="./data/train", transform=transform)
val_dataset = ImageFolder(root="./data/val", transform=transform)

#defining data loaders
train_dataloader = DataLoader(train_dataset, batch_size=150, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=2)

loss = nn.functional.cross_entropy

optimizer =optim.RMSprop(model.parameters(), lr=lr)
model = model.to(device)

loss_epoch = []
loss_val_epoch = []
for epoch in range(10):

    loss_train = 0
    loss_val_ep = 0
    for i, input in enumerate(traindataloader,0):
        data, label = input
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        y = model(data)
        loss_val = loss(y, label)
        loss_train += loss_val.item()
        
        loss_val.backward()
        optimizer.step()
    loss_train /= 150
    loss_epoch.append(loss_train)
    print("Train Epoch: {}, \t Loss: {}".format(epoch, loss_train))
    

    for i, input in enumerate(testdataloader,0):
        data, label = input
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        y = model(data)
        loss_val = loss(y, label)
        loss_val_ep += loss_val.item()

    loss_val_ep /= 100
    loss_val_epoch.append(loss_val_ep)

    print("Val Epoch: {}, \t Loss: {}".format(epoch, loss_val_ep))


plt.figure()
plt.plot(loss_epoch, label='Train loss')
plt.plot(loss_val_epoch, label='Val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')


model = model.to("cpu")
layers = list(model.children())[:-1]
# print(layers)
feature_extractor = nn.Sequential(*layers)


layers = list(model.children())[:-1]
feature_extractor = nn.Sequential(*layers)

save_features(feature_extractor, folder_path='./Images', train_name = 'features_train_updated', test_name = 'features_test_updated' )


# features_train_csv = pd.read_csv('features_train_updated.csv').values
# features_test_csv = pd.read_csv('features_test_updated.csv').values


# features_test = np.load('features_test_updated.npy')
# features_train = np.load('features_train_updated.npy')

# print(features_train.shape, features_test.shape)


train_features = []
train_labels = []

for batch_idx, (image, label) in enumerate(traindataloader):
    features = extract_features(image, feature_extractor)
    train_features.append(features)
    label = label.numpy()
    train_labels.append(label)

test_features = []
test_labels = []

for batch_idx, (image, label) in enumerate(testdataloader):
    features = extract_features(image, feature_extractor)
    test_features.append(features)
    label = label.numpy()
    test_labels.append(label)

train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(-1)

test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1)
print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)


problem1(train_features, train_labels, test_features, test_labels)


# problem1(train_features, train_labels, test_features, test_labels)