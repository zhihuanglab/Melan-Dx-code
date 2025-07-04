# data_processor.py
from typing import List, Dict, Optional, Set, Tuple
import os
import json
from PIL import Image
from dataclasses import dataclass
import logging
from pathlib import Path
from data_processing.data_statistics import DatasetAnalyzer



@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_data_path: str
    val_data_path: str
    test_data_path: str
    knowledge_data_path: str
    root_train_path: str
    root_val_path: str
    root_test_path: str
    max_pixel_size: int = 4096
    min_images_per_disease: int = 3
    required_disease_level: int = 4
    enforce_test_train_disease_match: bool = True
    output_distribution: bool = True
    distribution_output_dir: str = "stats"
    metadata_output_dir: str = "metadata"
    metadata_filename: str = "dataset_metadata.json"
    force_preprocess: bool = False
    save_hierarchy_info: bool = True


@dataclass
class ImageMetadata:
    path: str
    disease_name: str
    parent_name: str
    image_size: Optional[Tuple[int, int]] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "disease_name": self.disease_name,
            "parent_name": self.parent_name,
            "image_size": self.image_size
        }


@dataclass
class DatasetMetadata:
    """Store dataset metadata information"""
    train_images: List[ImageMetadata]
    val_images: List[ImageMetadata]
    test_images: List[ImageMetadata]
    knowledge_texts: List[str]
    knowledge_disease_names: List[str]
    disease_to_idx: Dict[str, int]
    filtered_diseases: Set[str]
    disease_to_parent: Dict[str, str]
    parent_to_grandparent: Dict[str, str]

    def save(self, filepath: str):
        """Save metadata as JSON"""
        metadata_dict = {
            "train_images": [img.to_dict() for img in self.train_images],
            "val_images": [img.to_dict() for img in self.val_images],
            "test_images": [img.to_dict() for img in self.test_images],
            "knowledge_texts": self.knowledge_texts,
            "knowledge_disease_names": list(self.knowledge_disease_names),
            "disease_to_idx": self.disease_to_idx,
            "filtered_diseases": list(self.filtered_diseases),
            "disease_to_parent": self.disease_to_parent,
            "parent_to_grandparent": self.parent_to_grandparent
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'DatasetMetadata':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            train_images=[ImageMetadata(**img) for img in data["train_images"]],
            val_images=[ImageMetadata(**img) for img in data["val_images"]],
            test_images=[ImageMetadata(**img) for img in data["test_images"]],
            knowledge_texts=data["knowledge_texts"],
            knowledge_disease_names=data["knowledge_disease_names"],
            disease_to_idx=data["disease_to_idx"],
            filtered_diseases=set(data["filtered_diseases"]),
            disease_to_parent=data["disease_to_parent"],
            parent_to_grandparent=data["parent_to_grandparent"]
        )

    @property
    def train_disease_names(self) -> List[str]:
        return [img.disease_name for img in self.train_images]

    @property
    def val_disease_names(self) -> List[str]:
        return [img.disease_name for img in self.val_images]

    @property
    def test_disease_names(self) -> List[str]:
        return [img.disease_name for img in self.test_images]


class DataProcessor:

    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metadata: Optional[DatasetMetadata] = None
        
        metadata_path = os.path.join(config.metadata_output_dir, config.metadata_filename)
        
        if config.force_preprocess:
            self.logger.info("Processing data from scratch...")
            self._process_and_save_metadata()
        else:
            self.logger.info(f"Loading existing metadata from {metadata_path}")
            self.metadata = DatasetMetadata.load(metadata_path)
            self._validate_loaded_metadata()
        
        if self.config.output_distribution:
            self.generate_distribution_stats()

    def _validate_loaded_metadata(self):
        """Validate loaded metadata"""
        if not self.metadata:
            raise ValueError("Metadata is None")
            
        # Validate required attributes and data integrity
        required_attrs = [
            'train_images', 'test_images', 'knowledge_texts',
            'knowledge_disease_names', 'disease_to_idx', 'filtered_diseases'
        ]
        for attr in required_attrs:
            if not hasattr(self.metadata, attr):
                raise ValueError(f"Missing required attribute: {attr}")
        
        # Validate data integrity
        if not self.metadata.train_images or not self.metadata.knowledge_texts:
            raise ValueError("Missing training images or knowledge texts")
        
        # Validate file paths and disease mapping
        train_diseases = {img.disease_name for img in self.metadata.train_images}
        if not all(d in self.metadata.disease_to_idx for d in train_diseases):
            raise ValueError("Invalid disease mapping in metadata")

    def _process_and_save_metadata(self):
        """Process and save metadata"""
        try:
            # Load training data
            train_images = self._load_image_metadata(
                self.config.train_data_path,
                self.config.root_train_path,
                dataset_type='train'
            )
            
            # Load validation data
            val_images = self._load_image_metadata(
                self.config.val_data_path,
                self.config.root_val_path,
                dataset_type='val'
            )
            
            # Load test data
            test_images = self._load_image_metadata(
                self.config.test_data_path,
                self.config.root_test_path,
                dataset_type='test'
            )
            
            # Get disease sets from all datasets
            train_diseases = {img.disease_name for img in train_images}
            val_diseases = {img.disease_name for img in val_images}
            test_diseases = {img.disease_name for img in test_images}
            
            # Check for diseases in training set that are missing from validation and test sets
            missing_val_diseases = train_diseases - val_diseases
            missing_test_diseases = train_diseases - test_diseases
            
            if missing_val_diseases:
                self.logger.warning("The following diseases exist in training set but have no images in validation set:")
                for disease in sorted(missing_val_diseases):
                    self.logger.warning(f"  - {disease}")
                    
            if missing_test_diseases:
                self.logger.warning("The following diseases exist in training set but have no images in test set:")
                for disease in sorted(missing_test_diseases):
                    self.logger.warning(f"  - {disease}")
            
            if self.config.enforce_test_train_disease_match:
                filtered_diseases = train_diseases
                val_images = [img for img in val_images if img.disease_name in filtered_diseases]
                test_images = [img for img in test_images if img.disease_name in filtered_diseases]
            else:
                filtered_diseases = train_diseases | val_diseases | test_diseases
            
            knowledge_texts, knowledge_disease_names = self._load_knowledge_metadata(filtered_diseases)
            
            self._validate_knowledge_coverage(filtered_diseases, set(knowledge_disease_names))
            
            unique_diseases = sorted(filtered_diseases)
            disease_to_idx = {name: idx for idx, name in enumerate(unique_diseases)}
            
            self.metadata = DatasetMetadata(
                train_images=train_images,
                val_images=val_images,
                test_images=test_images,
                knowledge_texts=knowledge_texts,
                knowledge_disease_names=knowledge_disease_names,
                disease_to_idx=disease_to_idx,
                filtered_diseases=filtered_diseases,
                disease_to_parent=self.disease_to_parent,
                parent_to_grandparent=self.parent_to_grandparent
            )
            
            os.makedirs(self.config.metadata_output_dir, exist_ok=True)
            metadata_path = os.path.join(
                self.config.metadata_output_dir, 
                self.config.metadata_filename
            )
            self.metadata.save(metadata_path)
            self.logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def _load_image_metadata(self, json_path: str, root_path: str, dataset_type: str = 'train') -> List[ImageMetadata]:

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_metadata = []
        root = root_path.replace("\\", "/")

        level_4_diseases = [
            d for d in data
            if d.get("level_number") == self.config.required_disease_level
               and len(d.get("images", [])) >= self.config.min_images_per_disease
        ]

        self.disease_to_parent = {}
        self.parent_to_grandparent = {}
        
        for disease in level_4_diseases:
            disease_name = disease.get('disease_name')
            parent_name = disease.get('parent')
            self.disease_to_parent[disease_name] = parent_name
            

            for d in data:
                if d.get('disease_name') == parent_name:
                    grandparent_name = d.get('parent')
                    if grandparent_name:
                        self.parent_to_grandparent[parent_name] = grandparent_name
                    break


        self.logger.info("Disease Hierarchy Structure:")
        for disease_name, parent_name in self.disease_to_parent.items():
            grandparent_name = self.parent_to_grandparent.get(parent_name, "No Grandparent")
            self.logger.info(f"Disease: {disease_name} -> Parent: {parent_name} -> Grandparent: {grandparent_name}")


        image_key = {
            'train': 'images',
            'val': 'val_images',
            'test': 'test_images'
        }.get(dataset_type, 'images')

        for disease in level_4_diseases:
            disease_name = disease.get('disease_name')
            parent_name = disease.get('parent')
            images = disease.get(image_key, [])

            for img_path in images:
                full_path = os.path.join(root, img_path).replace("\\", "/")
                if self._is_valid_image(full_path):
                    filtered_metadata.append(ImageMetadata(
                        full_path, 
                        disease_name,
                        parent_name
                    ))
                else:
                    self.logger.warning(f"Error process image too large {img_path}: {str(disease_name)}")

        return filtered_metadata

    def _is_valid_image(self, image_path: str) -> bool:

        try:
            with Image.open(image_path) as img:
                return max(img.size) <= self.config.max_pixel_size
        except Exception as e:
            self.logger.warning(f"Error validating image {image_path}: {str(e)}")
            return False

    def _load_knowledge_metadata(self, filtered_diseases: Set[str]) -> Tuple[List[str], List[str]]:

        try:
            with open(self.config.knowledge_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            texts = []
            disease_names = []

            for entry in data:
                disease_name = entry.get('disease_name')
                # Only process diseases in filtered_diseases
                if not disease_name or disease_name not in filtered_diseases:
                    continue

                for key, value in entry.items():
                    if key in ['images',
                                   'disease_name',  
    # 'Histopathology', 
    # 'Macroscopic appearance', 
    # 'Essential and desirable diagnostic criteria',  
    # "Definition", 
    "Related terminology", 
    "Subtype(s)", 
    "Localization", 
    "Clinical features",
    "Cytology",
    "Diagnostic molecular pathology",

    ]:
                        continue

                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                texts.append(item)
                                disease_names.append(disease_name)
                    elif isinstance(value, str) and value.strip():
                        texts.append(value)
                        disease_names.append(disease_name)

            if not texts:
                raise ValueError("No valid knowledge texts found for the filtered diseases")

            return texts, disease_names

        except Exception as e:
            self.logger.error(f"Error loading knowledge data: {str(e)}")
            raise

    def _validate_knowledge_coverage(self, filtered_diseases: Set[str], knowledge_diseases: Set[str]):
        missing_diseases = filtered_diseases - knowledge_diseases
        if missing_diseases:
            missing_list = ", ".join(sorted(missing_diseases))
            error_msg = f"Missing knowledge data for diseases: {missing_list}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def generate_distribution_stats(self):
        if not self.metadata:
            return

        analyzer = DatasetAnalyzer(self.config.distribution_output_dir)

        config_dict = {
            "required_disease_level": self.config.required_disease_level,
            "min_images_per_disease": self.config.min_images_per_disease,
            "enforce_test_train_disease_match": self.config.enforce_test_train_disease_match,
            "max_pixel_size": self.config.max_pixel_size
        }

        distribution = analyzer.generate_distribution_stats(
            self.metadata.train_disease_names,
            self.metadata.val_disease_names,
            self.metadata.test_disease_names,
            self.metadata.knowledge_disease_names,
            config_dict
        )

        analyzer.save_distribution_stats(distribution)

    @property
    def train_paths(self) -> List[str]:
        return [img.path for img in self.metadata.train_images]

    @property
    def test_paths(self) -> List[str]:
        return [img.path for img in self.metadata.test_images]

    def get_disease_index(self, disease_name: str) -> Optional[int]:
        return self.metadata.disease_to_idx.get(disease_name)

    def get_num_classes(self) -> int:
        return len(self.metadata.disease_to_idx)
