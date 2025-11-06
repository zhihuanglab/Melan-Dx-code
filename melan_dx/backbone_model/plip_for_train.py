import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import PIL
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset, Image
import logging
from PIL import Image


class PLIP:


    def __init__(self, model_name, auth_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess, self.model_hash = self._load_model(model_name, auth_token=auth_token)
        self.model = self.model.to(self.device)
        self.logger = logging.getLogger(__name__)


    def _load_model(self,
                    name: str,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    auth_token=None):

        model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
        preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)

        return model, preprocessing, hash

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):

        num_images = len(images)
        # print(self.model.device)
        num_batches = (num_images + batch_size - 1) // batch_size
        image_embeddings = []


        pbar = tqdm(
            total=num_batches,
            desc="Encoding images",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        for i in range(0, num_images, batch_size):

            batch_slice = slice(i, min(i + batch_size, num_images))
            batch_images = images[batch_slice]
            
            if isinstance(batch_images[0], str):
                loaded_images = []
                for img_path in batch_images:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        loaded_images.append(img)
                    except Exception as e:
                        self.logger.warning(f"Error loading image {img_path}: {str(e)}")
                        loaded_images.append(Image.new('RGB', (224, 224)))
                batch_images = loaded_images

            preprocessed = self.preprocess(
                images=batch_images,
                return_tensors='pt'
            )

            preprocessed = {k: v.to(self.device) for k, v in preprocessed.items()}

            batch_embeddings = self.model.get_image_features(**preprocessed)
            image_embeddings.append(batch_embeddings)

            pbar.update(1)

        pbar.close()
        return torch.cat(image_embeddings, dim=0)

    def encode_text(self, text: List[str], batch_size: int):

        num_texts = len(text)
        num_batches = (num_texts + batch_size - 1) // batch_size
        text_embeddings = []

        
        pbar = tqdm(
            total=num_batches,
            desc="Encoding texts",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        for i in range(0, num_texts, batch_size):

            batch_texts = text[i:i + batch_size]

            preprocessed = self.preprocess(
                text=batch_texts,
                return_tensors="pt",
                max_length=77,
                padding="max_length",
                truncation=True
            )

            preprocessed = {k: v.to(self.device) for k, v in preprocessed.items()}

            batch_embeddings = self.model.get_text_features(**preprocessed)
            text_embeddings.append(batch_embeddings)

            pbar.update(1)

        pbar.close()
        return torch.cat(text_embeddings, dim=0)


