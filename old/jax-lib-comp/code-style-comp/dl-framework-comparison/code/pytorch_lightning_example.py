import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import torch.nn as nn


class PetClassifier(pl.LightningModule):
	def __init__(self, num_classes, learning_rate=0.01):
		super(PetClassifier, self).__init__()
		self.loss_fn = torch.nn.CrossEntropyLoss()
		self.learning_rate = learning_rate
		
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
		self.dense1 = nn.Linear(128 * 28 * 28, 256)  # Adjust depending on input size
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

	def training_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self(inputs)
		loss = self.loss_fn(outputs, labels)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self(inputs)
		loss = self.loss_fn(outputs, labels)
		acc = (outputs.argmax(dim=1) == labels).float().mean()
		self.log('val_acc', acc)
		self.log('val_loss', loss)

	def test_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self(inputs)
		loss = self.loss_fn(outputs, labels)
		acc = (outputs.argmax(dim=1) == labels).float().mean()
		self.log('test_acc', acc)
		self.log('test_loss', loss)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return [optimizer]
	
	class PetDataModule(pl.LightningDataModule):
	def __init__(self, batch_size=32, train_percent=0.8):
		super().__init__()
		self.batch_size = batch_size
		self.train_percent = train_percent
		self.transform = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def prepare_data(self):
		pass
#		 torchvision.datasets.Imagenette('./data_img', split='train', transform=self.transform, size="320px", download=True)
#		 torchvision.datasets.Imagenette('./data_img', split='val', transform=self.transform, size="320px", download=True)

	def setup(self, stage=None):
		dataset = torchvision.datasets.Imagenette('./data_img', split='train', transform=self.transform, size="320px")
		train_size = int(len(dataset) * self.train_percent)
		val_size = len(dataset) - train_size
		self.train_set, self.val_set = random_split(dataset, [train_size, val_size])
		self.test_set = torchvision.datasets.Imagenette('./data_img', split='val', transform=self.transform, size="320px")

	def train_dataloader(self):
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_set, batch_size=self.batch_size)

	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size)
	

	data_module = PetDataModule(batch_size=512)

num_classes = 10  # Assuming OxfordIIITPet has 37 classes
model = PetClassifier(num_classes=num_classes, learning_rate=0.001)

trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1000)
start_time = datetime.now()
trainer.fit(model, data_module)

end_time = datetime.now()
print("Whole training took:", (end_time - start_time).seconds)
