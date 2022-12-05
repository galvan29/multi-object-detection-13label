import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torch.optim as optim
import time

def get_num_correct(preds, labels):
  return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

turno = True
class Network(nn.Module):
    def __init__(self, faseFinale):
        super(Network, self).__init__()
        self.faseFinale = faseFinale
        # CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, stride=2, kernel_size=7, padding = 4),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=7, padding = 'same'),
            nn.BatchNorm2d(12))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels= 25, kernel_size=3, padding = 3), nn.BatchNorm2d(25))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=25, out_channels= 75, kernel_size=3, padding = 2), nn.BatchNorm2d(75))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels= 150, kernel_size=3, padding = 2),
            nn.Conv2d(in_channels=150, out_channels= 150, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(150))

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=150, out_channels= 350, kernel_size=3, padding = 2), nn.BatchNorm2d(350))
        
        self.conv6 = nn.Conv2d(in_channels=350, out_channels= 350, kernel_size=3, padding = 1)

        self.conv7 = nn.Conv2d(in_channels=350, out_channels= 1024, kernel_size=3)

        self.class_fc0 = nn.Linear(in_features=1024, out_features=750)
        self.class_fc0_1 = nn.Linear(in_features=750, out_features=512)
        self.class_fc1 = nn.Linear(in_features=512, out_features=300)
        self.class_fc1_2 = nn.Linear(in_features=300, out_features=150)


        self.class_fc2 = nn.Linear(in_features=150, out_features=20)
        self.class_out = nn.Linear(in_features=20, out_features=13)
		
        self.class_fc2a = nn.Linear(in_features=150, out_features=20)
        self.class_outa = nn.Linear(in_features=20, out_features=13)


        self.box_fc0 = nn.Linear(in_features=1024, out_features=750)

        self.box_fc0_1 = nn.Linear(in_features=750, out_features=512)
        self.box_fc1 = nn.Linear(in_features=512, out_features=300)
        self.box_fc1_2 = nn.Linear(in_features=300, out_features=150)
        self.box_fc2 = nn.Linear(in_features=150, out_features=20)
        self.box_out = nn.Linear(in_features=20, out_features=4)
		
        self.box_fc0_1b = nn.Linear(in_features=750, out_features=512)
        self.box_fc1b = nn.Linear(in_features=512, out_features=300)
        self.box_fc1_2b = nn.Linear(in_features=300, out_features=150)
        self.box_fc2b = nn.Linear(in_features=150, out_features=20)
        self.box_outb = nn.Linear(in_features=20, out_features=4)

        self.relu = nn.LeakyReLU(inplace = True, negative_slope = 0.01)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(0.3)
        self.drop2d = nn.Dropout2d(0.1)

    def forward(self, t):
        global turno
        t = self.conv1(t)
        t = self.relu(t)
        t = self.max_pool(t)
        t = self.drop2d(t)

        t = self.conv2(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv3(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv4(t)
        t = self.relu(t)
        t = self.max_pool(t)
        t = self.drop2d(t)

        t = self.conv5(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv6(t) 

        t = self.conv7(t) 

        t = torch.flatten(t, start_dim=1)

        class_t = self.class_fc0(t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc0_1(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc1(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc1_2(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)
        
        box_t = self.box_fc0(t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)
        

        class_t1 = self.class_fc2(class_t)
        class_t1 = self.relu(class_t1)
        class_t1 = self.drop(class_t1)
        class_t1 = F.softmax(self.class_out(class_t1), dim=1)

        box_t0 = self.box_fc0_1(box_t)
        box_t0 = self.relu(box_t0)
        box_t0 = self.drop(box_t0)

        box_t0 = self.box_fc1(box_t0)
        box_t0 = self.relu(box_t0)
        box_t0 = self.drop(box_t0)

        box_t0 = self.box_fc1_2(box_t0)
        box_t0 = self.relu(box_t0)
        box_t0 = self.drop(box_t0)

        box_t0 = self.box_fc2(box_t0)
        box_t0 = self.relu(box_t0)
        box_t0 = self.drop(box_t0)

        box_t0 = self.box_out(box_t0)
        box_t0 = F.sigmoid(box_t0)


        class_t2 = self.class_fc2a(class_t)
        class_t2 = self.relu(class_t2)
        class_t2 = self.drop(class_t2)
        class_t2 = self.class_outa(class_t2)
        class_t2 = F.softmax(class_t2, dim=1)

        box_t1 = self.box_fc0_1b(box_t)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_fc1b(box_t1)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_fc1_2b(box_t1)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_fc2b(box_t1)
        box_t1 = self.relu(box_t1)
        box_t1 = self.drop(box_t1)

        box_t1 = self.box_outb(box_t1)
        box_t1 = F.sigmoid(box_t1)

        return [class_t1, box_t0, class_t2, box_t1]

def initialize_weights(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

def train(num_of_epochs, lr, dataset, valdataset, samples, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=True)

    model = Network(False)
    model.apply(initialize_weights)
    model = model.to(device)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    epochs = []
    losses = []

    # Creating a directory for storing models
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    for epoch in range(num_of_epochs):
        tot_loss = 0
        tot_correct = 0
        train_start = time.time()
        model.train()

        for batch, (x, y, z) in enumerate(dataloader):
        	# Converting data from cpu to GPU if available to improve speed
            x,y1,z1 = x.to(device),y[:,0].to(device),z[:,0].to(device)
            
            optimizer.zero_grad()
            [y_pred, z_pred, y_pred1, z_pred1]= model(x)
            class_loss = 0
            box_loss = 0
            class_loss = F.cross_entropy(y_pred, y1)
            box_loss = F.mse_loss(z_pred, z1)

            (class_loss+box_loss).backward(retain_graph=True)
            optimizer.step()
            [y_pred, z_pred, y_pred1, z_pred1]= model(x)

            y2,z2 = y[:,1].to(device),z[:,1].to(device)

            optimizer.zero_grad()
            class_loss = F.cross_entropy(y_pred1, y2)
            box_loss = F.mse_loss(z_pred1, z2)

            (class_loss+box_loss).backward()
            optimizer.step()

            print("Train batch:", batch+1, " epoch: ", epoch, " ", (time.time()-train_start)/60, end='\r')

        model.eval()

        for batch, (x, y, z) in enumerate(valdataloader):
            x,y,z = x.to(device),y[:,0].to(device),z[:,0].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                [y_pred, z_pred, y_pred1, z_pred1]= model(x)
                class_loss = F.cross_entropy(y_pred, y)
                box_loss = F.mse_loss(z_pred, z)

            tot_loss += (class_loss.item() + box_loss.item())
            tot_correct += get_num_correct(y_pred, y)

            print("Test batch:", batch+1, " epoch: ", epoch, " ", (time.time()-train_start)/60, end='\r')

        epochs.append(epoch)
        losses.append(tot_loss)

        print("Epoch", epoch, "Accuracy", tot_correct/3000, "loss:",
              tot_loss/3000, " time: ", (time.time()-train_start)/60, " mins")
        if epoch%40 == 0:
            torch.save(model.state_dict(), savedir+"/model_ep"+str(epoch+1)+".pth")