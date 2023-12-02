import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import cv2


from torchvision import datasets
from torch.utils.data import Dataset

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
traindataset = LoadDataset('Images/train')
testdataset = LoadDataset('Images/test')

from torch.utils.data import DataLoader

# Create dataloader
traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
testdataloader =  DataLoader(testdataset, batch_size=1, shuffle=False)


# Load the pre-trained ResNet model
model = models.resnet18(weights="IMAGENET1K_V1")

# Modify the model to remove the last fully connected layer (classifier)
model = torch.nn.Sequential(*(list(model.children())[:-1]))



#function to extract features
def extract_features(image, model):
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()




# Now we run a loop to create a list of train features and its labels
train_features = []
train_labels = []

for batch_idx, (image, label) in enumerate(traindataloader):
    features = extract_features(image, model)
    train_features.append(features)
    label = label.numpy()
    train_labels.append(label)



# Similarly we can do for list of test features
test_features = []
test_labels = []

for batch_idx, (image, label) in enumerate(testdataloader):
    features = extract_features(image, model)
    test_features.append(features)
    label = label.numpy()
    test_labels.append(label)



# convert into numpy arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_labels = train_labels.reshape(-1)

test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1)


# Initialize KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
# Fit the KNN model
knn.fit(train_features, train_labels)


# let us make predictions using this KNN
preds = knn.predict(test_features)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

c_matrix = confusion_matrix(test_labels,preds)
disp = ConfusionMatrixDisplay(c_matrix)
disp.plot()

confusion = c_matrix
num_classes = len(c_matrix)
TP = [confusion[i, i] for i in range(num_classes)]
TN = [np.sum(confusion) - np.sum(confusion[i, :]) - np.sum(confusion[:, i]) + confusion[i, i] for i in range(num_classes)]
FP = [np.sum(confusion[:, i]) - confusion[i, i] for i in range(num_classes)]
FN = [np.sum(confusion[i, :]) - confusion[i, i] for i in range(num_classes)]

TP = np.array(TP)
TN = np.array(TN)
FP = np.array(FP)
FN = np.array(FN)
print(TP, TN, FP, FN)


#The accuracy of the classes
acc = (TP+TN)/(TP+FP+FN+TN)
print(acc)

#The sensitivity of the classes
sens = TP/(TP+FN)
print(sens)

#The specificity of the classes
spec = TN/(TN+FP)
print(spec)





########################## Fine Tuning #########################



EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


from torch.utils.data import Subset
import random

# create seperate traindataloader and validataloader
all_dataset = LoadDataset('Images/train')

# Split training and validation into 120/30
train_indices = random.sample(range(150), 120)
valid_indices = list(set(range(150))- set(train_indices))

# Create dataset
train_dataset = Subset(all_dataset, train_indices)
valid_dataset = Subset(all_dataset, valid_indices)

# Create dataloader
traindataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
validataloader = DataLoader(valid_dataset, batch_size= BATCH_SIZE, shuffle=True)

model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze the model weights
for param in model.parameters():
    param.requires_grad=False

# Add an FC layer at the end to match the number of classes in our dataset
model.fc = torch.nn.Linear(model.fc.in_features, 5)


# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)



# Train the model
from datetime import datetime
from torch.autograd import Variable

# # Create directory to save the model states
current_datetime = datetime.now()
date_time_format = current_datetime.strftime("%m-%d_%H-%M")
directory_name = f"weights/run_{date_time_format}"

# Create the directory
os.makedirs(directory_name)
print(f"Model states saved in :{directory_name}")

train_loss = []
train_accu = []
val_loss = []
val_accu = []

for e in range(EPOCHS):
    print("EPOCH: {}/{}".format(e+1, EPOCHS)) 
    running_loss = 0
    correct_pred = 0
    total_pred = 0
    #set the model to train
    model.train()
    epoch_train_loss = []
    epoch_test_accu = []
    for batch_idx, (image, label) in enumerate(traindataloader):
        x = Variable(image)
        y = Variable(label)
        #forward pass
        optimizer.zero_grad()
        pred = model.forward(x)
        #calculate loss
        loss = criterion(pred,y)
        #calculate accuracy
        _, mpred = torch.max(pred,1)
        correct_pred += (mpred == label).sum().item()
        total_pred += label.size(0)
        #backpropagate the loss
        loss.backward()
        optimizer.step()
        #detach
        loss = loss.detach().numpy()
        running_loss += loss      
    train_loss.append(running_loss/len(traindataloader))
    train_accu.append(correct_pred/total_pred)
    print("Train loss: {:.5f}, Train accuracy: {:.5f}".format(
        running_loss/len(traindataloader),correct_pred/total_pred))

    #set the model to test
    model.eval()
    running_loss = 0
    correct_pred = 0
    total_pred = 0
    for batch_idx, (image, label) in enumerate(validataloader):
        x = Variable(image)
        y = Variable(label)
        #forward pass
        pred = model.forward(x)
        #calculate loss
        loss = criterion(pred, y)
        #calculate accuracy
        _, mpred = torch.max(pred,1)
        correct_pred += (mpred == label).sum().item()
        total_pred += label.size(0)
        #detach
        loss = loss.detach().numpy()
        running_loss += loss
    val_loss.append(running_loss/len(validataloader))
    val_accu.append(correct_pred/total_pred)
    print("Validation loss: {:.5f}, Validation accuracy: {:.5f}".format(
        running_loss/len(validataloader),correct_pred/total_pred))
    print('-------------')

    # save model after every 10 epochs
    if e%10==0:
        model_weights_name = f"epoch_{e}.pth"
        model_weights_name = os.path.join(directory_name, model_weights_name)
        torch.save(model.state_dict(), model_weights_name)




import matplotlib.pyplot as plt

no_epochs = np.arange(len(train_loss))
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.title('Training Curve for Loss')
plt.xlabel('No of Epochs')
plt.ylabel('Cross Entropy Loss')
plt.plot(no_epochs, train_loss, label='Training Loss')
plt.plot(no_epochs, val_loss, label='Validation Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy for the Model')
plt.xlabel('No of Epochs')
plt.ylabel('Accuracy')
plt.plot(no_epochs, train_accu, label='Training Accuracy')
plt.plot(no_epochs, val_accu, label='Validation Accuracy')
plt.legend()


# Let us now perform inference on test set
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)

#load model from epoch
epoch_to_load = 80

# load model weights
model.load_state_dict(torch.load(f"{directory_name}/epoch_{epoch_to_load}.pth"))




# lists to save the values
labels_list = []
preds_list = []

# inference on test data
model.eval()
running_loss = 0
correct_pred = 0
total_pred = 0
for batch_idx, (image, label) in enumerate(testdataloader):
    x = Variable(image)
    y = Variable(label)
    pred = model.forward(x)
    #calculate loss 
    loss = criterion(pred, y)
    #calculate accuracy
    _, mpred = torch.max(pred,1)
    correct_pred += (mpred == label).sum().item()
    total_pred += label.size(0)
    #detach
    loss = loss.detach().numpy()
    running_loss += loss
    #saving values
    mpred = mpred.detach().numpy()
    label = label.detach().numpy()
    labels_list.append(label)
    preds_list.append(mpred)

print('Test Accuracy:', correct_pred/total_pred)
print('Test loss:', running_loss/len(testdataloader))


#create a confusion matrxi
labels_list = np.array(labels_list)
preds_list = np.array(preds_list)


# confusion matrix of the predictions
c_matrix = confusion_matrix(labels_list,preds_list)
disp = ConfusionMatrixDisplay(c_matrix)
disp.plot()