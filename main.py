import os
import random
from images_extraction import ImagesDataset
from PIL import Image
import numpy as np
import random
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import random_split, TensorDataset , DataLoader
from arcitecture import MyCNN
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import torchvision.transforms as transforms

show_progress = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


directory = r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\JKU\Semester 2\Python 2\Unit 7 (Project)\training_data"



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a specific seed
set_seed(42)


##############Transforms
data_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])



# Initialize the ImagesDataset

resized_images = []
classids = []
classnames = []

model = MyCNN()


dataset = ImagesDataset(directory, 100, 100,dtype=int)

#for resized_image,classid,classname,_ in dataset:

 #    resized_images.append(resized_image)
  #   classids.append(classid
    # classnames.append(classname)

#images_tensor = torch.stack(resized_images)
#target_tensor = torch.tensor(classids)
#all_dataset = TensorDataset(images_tensor,target_tensor)
##################################################

proportions = [.70, .20, .10]
lengths = [int(p * len(dataset)) for p in proportions]
lengths[-1] = len(dataset) - sum(lengths[:-1])
train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)
###################################################################



class model_train():




  @staticmethod
  def train_and_validate(model_class, train_dataset, valid_dataset, learning_rates, optimizer_class, batch_sizes, epochs, device,best_model_path, show_progress=True,patience=5):

                   best_loss = float('inf')
                   best_hyperparams = None
                   #best_model_path = 'model.pth'
                   device = torch.device(device if torch.cuda.is_available() else 'cpu')
                 #  model = MyCNN()
                   epochs_without_improving = 0

                   #for lr in learning_rates:
                    #   for optimizer_class in optimizers:
                     #      for batch_size in batch_sizes:
                            #    print(f"Training with Learning Rate = {lr}, Optimizer = {optimizer_class.__name__}, Batch Size = {batch_size} on device {device}")
                   #optimizer_class = torch.optim.Adam
                   train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True)
                   validation_loader = DataLoader(valid_dataset, batch_size=batch_sizes, shuffle=False)
                              #  model = model_class().to(device)
                   optimizer = optimizer_class(model.parameters(), lr=learning_rates)
                   criterion = nn.CrossEntropyLoss()


                   if os.path.exists(best_model_path):
                       print("hatshtghal ya big boss")

                   else:
                       print("Shoof had ynekak msh hatshtghal")

                   epochs_range = range(epochs)
                   if show_progress:
                        epochs_range = tqdm(epochs_range, desc="Training Epochs")

                   for epoch in epochs_range:
                        model.train()
                        total_train_loss = 0

                        for x_train, y_train, classnames, filepaths in train_loader:
                         #   x_train, y_train = x_train.to(device), y_train.to(device)
                           # print(len(x_train),len(y_train),len(classnames),len(filepaths))


                            prediction = model(x_train)
                            train_loss = criterion(prediction, y_train)

                            train_loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                            total_train_loss += train_loss.item()  # loss per batch

                        average_train_loss = total_train_loss / len(train_loader)  # average train error per epoch(per number of batches)



                        with torch.no_grad():  # turns off gradient tracking, because we don't need them for validation
                            model.eval()
                            total_val_loss = 0
                            for x_eval, y_eval, _, _ in validation_loader:
                               # x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                                validation_prediction = model(x_eval)
                                validation_loss = criterion(validation_prediction, y_eval)

                                total_val_loss += validation_loss.item()
                            average_val_loss = total_val_loss / len(validation_loader)


                        print(f"Epoch: {epoch} --- Train loss: {average_train_loss:7.4f} --- Eval loss: {average_val_loss:7.4f}")
                        if average_val_loss < best_loss:
                               best_loss = average_val_loss
                               epochs_without_improving = 0
                          ##  best_hyperparams = (lr, optimizer_class, batch_size)
                            #   torch.save(model.state_dict(), best_model_path)
                          #  print(f"Model saved with loss:{best_loss:.4f}")
                        else:
                               epochs_without_improving += 1

                #        print(f"Best hyperparameters: Learning Rate = {learning_rates}, Optimizer = {optimizers}, Batch Size = {batch_sizes}")
                        #print(f"Best validation loss: {best_loss:.4f} with the hyperparamters:{best_hyperparams}")
                        #print("Model ")

                 #       if epochs_without_improving >= patience :
                  #             print("Early stopping happened")
                   #            break

                   torch.save(model.state_dict(),best_model_path)

                   if os.path.exists(best_model_path) :
                       print(f"Model saved at {best_model_path}")





          #Print the best hyperparameters and the corresponding loss
                    #print(f"Best hyperparameters: Learning Rate = {best_hyperparams[0]}, Optimizer = {best_hyperparams[1].__name__}, Batch Size = {best_hyperparams[2]}")
                    #print(f"Best validation loss: {best_loss:.4f}")


  @staticmethod
  def test(model_class, test_dataset, batch_size,model_path):


        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
       # model = model_class().to(device)
        model = MyCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_test_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for x_test, y_test, _, _ in test_loader:
                #x_test, y_test = x_test.to(device), y_test.to(device)
                test_prediction = model(x_test)
                test_loss = criterion(test_prediction, y_test)

                total_test_loss += test_loss.item()
                _, predicted = torch.max(test_prediction, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(y_test.cpu().numpy())

        average_test_loss = total_test_loss / len(test_loader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Test Loss: {average_test_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(classification_report(true_labels, predictions))



#learning_rates = [0.001, 0.01,0.1]
#optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]
#batch_sizes = [16, 32,64]
epochs = 23
#best_model_path = r"C:\Users\alime\OneDrive - Johannes Kepler Universität Linz\JKU\Semester 2\Python 2\Unit 7 (Project)\files\model.pth(VGG-11,64 epochs good model)"
#x = test(model, test_dataset, batch_size=64, model_path=best_model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_train.train_and_validate(model, train_dataset, valid_dataset, 0.001,torch.optim.Adam,16, epochs,best_model_path="model.pth",device=device,patience=7)
model_train.test(model, test_dataset,batch_size=16,model_path="model.pth")

