import torch
import torchvision
import torchvision.transforms as transforms

from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 128
TRAIN_PERCENT = 0.8
VAL_PERCENT = 1.0 - TRAIN_PERCENT
LR = 0.001
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
	transforms.Resize((224, 224)),  # Randomly crop and resize to 224x224
	transforms.ToTensor(),			  # Convert the image to a tensor
	# Normalize using ImageNet's mean and std
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets for training & validation, download if necessary
dataset = torchvision.datasets.Imagenette(
	'./data_img', split='train', transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, (int(
	len(dataset)*TRAIN_PERCENT), len(dataset) - int(len(dataset)*TRAIN_PERCENT)))
test_set = torchvision.datasets.Imagenette(
	'./data_img', split='val', transform=transform)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
	train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
	val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


class SimpleCNN(nn.Module):
	def __init__(self, num_classes=10):
		super(SimpleCNN, self).__init__()

		# Define layers
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.bn5 = nn.BatchNorm2d(128)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.flatten = nn.Flatten()
		# Assuming input size is 224x224
		self.dense1 = nn.Linear(128 * 28 * 28, 256)
		self.bn7 = nn.BatchNorm1d(256)
		self.relu = nn.ReLU()

		self.output_layer = nn.Linear(256, num_classes)

	def forward(self, x):
		# First convolutional block
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool1(x)

		# Second convolutional block
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		x = self.pool2(x)

		# Third convolutional block
		x = self.conv5(x)
		x = self.bn5(x)
		x = self.relu(x)
		x = self.pool3(x)

		# Flatten and dense layers
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.bn7(x)
		x = self.relu(x)

		# Output layer
		x = self.output_layer(x)
		return x


model = SimpleCNN()
model = model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index):
	running_loss = 0.
	last_loss = 0.
	for i, data in enumerate(training_loader):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		# scheduler.step()

	return running_loss/len(training_loader)


epoch_number = 0
model = model.to(device)
model = torch.jit.script(model)
for epoch in range(EPOCHS):
	if epoch == 1:
		start_time = datetime.now()
	model.train()
	running_loss = 0.
	last_loss = 0.
	train_acc = 0.0
	for data in training_loader:
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		train_acc += (outputs.argmax(dim=1) == labels).float().mean()

	avg_loss = running_loss/len(training_loader)
	train_acc = train_acc/len(training_loader)
	running_vloss = 0.0
	model.eval()
	val_acc = 0.0
	with torch.no_grad():
		for vdata in val_loader:
			vinputs, vlabels = vdata
			vinputs = vinputs.to(device)
			vlabels = vlabels.to(device)
			voutputs = model(vinputs)
			vloss = loss_fn(voutputs, vlabels)
			val_acc += (outputs.argmax(dim=1) == labels).float().mean()
			running_vloss += vloss
	running_vloss /= len(val_loader)
	val_acc /= len(val_loader)
	print('EPOCH {}: train_loss: {}, train_acc {} val_loss: {} val acc {}'.format(
		epoch_number + 1, avg_loss, train_acc, vloss, val_acc))

	epoch_number += 1

end_time = datetime.now()
print("Whole training took:", (end_time - start_time).seconds)
