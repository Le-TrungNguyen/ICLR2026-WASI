import logging
import os
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from .sampling import split_and_shuffle
from .train_test_split.train_test_split_cub200 import prepare_cub200
from .train_test_split.train_test_split_flowers102 import prepare_flowers102
from .train_test_split.train_test_split_pets import prepare_pets
from .train_test_split.TextDataset import TextDataset

from datasets import load_dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}


class ClsDataset(LightningDataModule):
    def __init__(self, data_dir, name='mnist',
                    train_split=0.8,
                    batch_size=32, train_shuffle=True,
                    width=224, height=224,
                    train_workers=4, val_workers=1, num_train_batch=None, num_val_batch=None, num_test_batch=None, max_length=512):
        super(ClsDataset, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.width = width
        self.height = height
        self.num_classes = 0
        self.num_train_batch = num_train_batch
        self.num_val_batch = num_val_batch
        self.num_test_batch = num_test_batch
        self.max_length = max_length


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def setup(self, stage):

        if self.name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.1307,), (0.3081,))])
            raw_train_dataset = datasets.MNIST(self.data_dir, train=True, download=True,
                                         transform=apply_transform)
            test_dataset = datasets.MNIST(self.data_dir, train=False, download=True,
                                        transform=apply_transform)
            self.num_classes = len(raw_train_dataset.classes)

        elif self.name == 'pets':
            prepare_pets(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = len(raw_train_dataset.classes)

        elif self.name == 'cub200':
            prepare_cub200(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = len(raw_train_dataset.classes)

        elif self.name == 'flowers102':
            prepare_flowers102(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            val_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "val"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = len(train_dataset.classes)

        elif self.name == 'cifar10':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True,
                                        transform=apply_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True,
                                        transform=apply_transform)
            self.num_classes = len(raw_train_dataset.classes)
        elif self.name == 'cifar100':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True,
                                            transform=apply_transform)
            test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True,
                                            transform=apply_transform)
            self.num_classes = len(raw_train_dataset.classes)
        elif self.name == 'imagenet' or self.name == 'mini_imagenet':
            apply_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.width),
                transforms.RandomHorizontalFlip(),      
                transforms.ToTensor(),                  
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            raw_train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=apply_transform)
            test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=apply_transform)
            self.num_classes = len(raw_train_dataset.classes)
        
        elif self.name == 'isic2018':
            apply_transform = transforms.Compose([
                transforms.Resize((self.width, self.height)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            raw_train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=apply_transform)
            test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=apply_transform)
            self.num_classes = len(raw_train_dataset.classes)

        elif self.name == 'BoolQ':
            # Load BoolQ dataset
            if not hasattr(self, 'tokenizer'):
                raise ValueError("Tokenizer must be set before loading BoolQ dataset. Call set_tokenizer() first.")

            dataset = load_dataset("super_glue", "boolq", trust_remote_code=True)
        elif self.name == 'AG_News':
            # Load AG_News dataset
            if not hasattr(self, 'tokenizer'):
                raise ValueError("Tokenizer must be set before loading AG_News dataset. Call set_tokenizer() first.")

            dataset = load_dataset("ag_news", trust_remote_code=True)
        
        elif self.name == 'c4':
            # Load C4 dataset
            if not hasattr(self, 'tokenizer'):
                raise ValueError("Tokenizer must be set before loading C4 dataset. Call set_tokenizer() first.")
            
            if self.num_train_batch is not None:
                train_samples = self.batch_size * self.num_train_batch
            else:
                train_samples = 100000  # Default
                
            if self.num_val_batch is not None:
                val_samples = self.batch_size * self.num_val_batch
            else:
                val_samples = 10000  # Default
            
            dataset = load_dataset(
                "c4", 
                "en",
                cache_dir="./data",
                trust_remote_code=True
            )

            # Load C4 dataset
            train_data = dataset["train"].select(range(train_samples))
            val_data = dataset["validation"].select(range(val_samples))

            # Tạo dataset objects
            self.train_dataset = TextDataset(
                train_data,
                self.tokenizer,
                max_length=self.max_length,
                task='lm'  # or 'cls' depend on task
            )
            
            self.val_dataset = TextDataset(
                val_data,
                self.tokenizer,
                max_length=self.max_length,
                task='lm'
            )
            
            self.test_dataset = self.val_dataset
            
            self.num_classes = self.tokenizer.vocab_size
        
        elif self.name == 'wikitext':
            train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            val_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

            self.train_dataset = TextDataset(
                train_data,
                self.tokenizer,
                max_length=self.max_length,
                task='lm'
            )
            
            self.val_dataset = TextDataset(
                val_data,
                self.tokenizer,
                max_length=self.max_length,
                task='lm'
            )
            
            self.test_dataset = self.val_dataset
            
            self.num_classes = self.tokenizer.vocab_size

        else:
            raise NotImplementedError

        excluded_names = ['flowers102', 'BoolQ', 'AG_News', 'c4', 'wikitext']

        if self.name not in excluded_names:
            idx_train, idx_val = split_and_shuffle(raw_train_dataset, self.train_split)
            if self.num_train_batch != None: idx_train = idx_train[:self.batch_size*self.num_train_batch]
            if self.num_val_batch != None: idx_val = idx_val[:self.batch_size*self.num_val_batch]
            
            self.train_dataset = DatasetSplit(raw_train_dataset, idx_train)
            self.val_dataset = DatasetSplit(raw_train_dataset, idx_val)

            if self.num_test_batch != None: self.test_dataset = DatasetSplit(test_dataset, np.arange(self.batch_size*self.num_test_batch))
            else: self.test_dataset = DatasetSplit(test_dataset, np.arange(len(test_dataset)))
        elif self.name == 'flowers102':
            if self.num_train_batch != None: 
                self.train_dataset = DatasetSplit(train_dataset, np.arange(self.batch_size*self.num_train_batch))
            else: self.train_dataset = DatasetSplit(train_dataset, np.arange(len(train_dataset)))
            
            if self.num_val_batch != None: self.val_dataset = DatasetSplit(val_dataset, np.arange(self.batch_size*self.num_val_batch))
            else: self.val_dataset = DatasetSplit(val_dataset, np.arange(len(val_dataset)))

            if self.num_test_batch != None: self.test_dataset = DatasetSplit(test_dataset, np.arange(self.batch_size*self.num_test_batch))
            else: self.test_dataset = DatasetSplit(test_dataset, np.arange(len(test_dataset)))
        elif self.name == 'BoolQ':
            train_data = dataset["train"]
            val_data = dataset["validation"]

            if self.num_train_batch is not None:
                num_train_samples = min(self.batch_size * self.num_train_batch, len(train_data))
                train_data = train_data.select(range(num_train_samples))

            if self.num_val_batch is not None:
                num_val_samples = min(self.batch_size * self.num_val_batch, len(val_data))
                val_data = val_data.select(range(num_val_samples))

            def preprocess_function(examples):
                inputs = [f"{q} [SEP] {p}" for q, p in zip(examples["question"], examples["passage"])]
                encodings = self.tokenizer(inputs, truncation=True, padding="max_length", max_length=self.max_length)

                encodings["label"] = [1 if ans else 0 for ans in examples["label"]]
                return encodings

            self.train_dataset = train_data.map(preprocess_function, batched=True)
            self.val_dataset = val_data.map(preprocess_function, batched=True)
            self.train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            self.val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        elif self.name =='AG_News':
            def preprocess_function(examples):
                return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_length)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            
            train_data = tokenized_dataset["train"]
            val_data = tokenized_dataset["test"]

            # Train dataset
            if self.num_train_batch is not None:
                num_train_samples = self.batch_size * self.num_train_batch
                if num_train_samples < len(train_data):
                    self.train_dataset = train_data.select(range(num_train_samples))
                else:
                    self.train_dataset = train_data
            else:
                self.train_dataset = train_data

            # Val dataset
            if self.num_val_batch is not None:
                num_val_samples = self.batch_size * self.num_val_batch
                if num_val_samples < len(val_data):
                    self.val_dataset = val_data.select(range(num_val_samples))
                else:
                    self.val_dataset = val_data
            else:
                self.val_dataset = val_data


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle,
            num_workers=self.train_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.val_workers, pin_memory=True, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)