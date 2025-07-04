# data_statistics.py
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Dict, List, Set
from datetime import datetime
from collections import Counter
import logging
from typing import List, Dict, Set


@dataclass
class ClassStats:
    """Per-class statistics"""
    count: int
    percentage: float


@dataclass
class DatasetStats:
    """Dataset statistics with disease counts"""
    total_count: int
    num_diseases: int
    class_statistics: Dict[str, ClassStats]
    timestamp: str = str(datetime.now())

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_count,
            "num_diseases": self.num_diseases,
            "class_statistics": {
                disease: {
                    "count": stats.count,
                    "percentage": round(stats.percentage, 2)
                }
                for disease, stats in self.class_statistics.items()
            },
            "timestamp": self.timestamp
        }


@dataclass
class CrossDatasetStats:
    """Statistics across datasets"""
    total_diseases: int
    common_diseases: List[str]
    train_only: List[str]
    val_only: List[str]
    test_only: List[str]
    knowledge_only: List[str]

    def to_dict(self) -> dict:
        return {
            "total_unique_diseases": self.total_diseases,
            "common_diseases": {
                "count": len(self.common_diseases),
                "diseases": sorted(self.common_diseases)
            },
            "train_only_diseases": {
                "count": len(self.train_only),
                "diseases": sorted(self.train_only)
            },
            "val_only_diseases": {
                "count": len(self.val_only),
                "diseases": sorted(self.val_only)
            },
            "test_only_diseases": {
                "count": len(self.test_only),
                "diseases": sorted(self.test_only)
            },
            "knowledge_only_diseases": {
                "count": len(self.knowledge_only),
                "diseases": sorted(self.knowledge_only)
            }
        }


@dataclass
class DatasetDistribution:
    """Complete distribution information for all datasets"""
    train_images: DatasetStats
    val_images: DatasetStats
    test_images: DatasetStats
    knowledge: DatasetStats
    cross_dataset_stats: CrossDatasetStats
    configuration: Dict[str, any]

    def to_dict(self) -> dict:
        return {
            "configuration": self.configuration,
            "cross_dataset_statistics": self.cross_dataset_stats.to_dict(),
            "training_set": self.train_images.to_dict(),
            "validation_set": self.val_images.to_dict(),
            "test_set": self.test_images.to_dict(),
            "knowledge_base": self.knowledge.to_dict()
        }


class DatasetAnalyzer:
    """Analyzes dataset distributions and generates statistics"""

    def __init__(self, output_dir: str = "stats"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def _calculate_dataset_stats(self, items: List[str]) -> DatasetStats:
        """Calculate distribution statistics for a dataset"""
        counts = Counter(items)
        total = len(items)
        num_diseases = len(counts)

        class_statistics = {}
        for disease, count in counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            class_statistics[disease] = ClassStats(count=count, percentage=percentage)

        return DatasetStats(
            total_count=total,
            num_diseases=num_diseases,
            class_statistics=class_statistics
        )

    def _calculate_cross_dataset_stats(
        self,
        train_diseases: Set[str],
        val_diseases: Set[str],
        test_diseases: Set[str],
        knowledge_diseases: Set[str]
    ) -> CrossDatasetStats:
        """Calculate statistics across all datasets"""
        common_diseases = train_diseases & val_diseases & test_diseases & knowledge_diseases
        
        other_sets = val_diseases | test_diseases | knowledge_diseases
        train_only = train_diseases - other_sets
        
        other_sets = train_diseases | test_diseases | knowledge_diseases
        val_only = val_diseases - other_sets
        
        other_sets = train_diseases | val_diseases | knowledge_diseases
        test_only = test_diseases - other_sets
        
        other_sets = train_diseases | val_diseases | test_diseases
        knowledge_only = knowledge_diseases - other_sets
        
        all_diseases = train_diseases | val_diseases | test_diseases | knowledge_diseases

        return CrossDatasetStats(
            total_diseases=len(all_diseases),
            common_diseases=list(common_diseases),
            train_only=list(train_only),
            val_only=list(val_only),
            test_only=list(test_only),
            knowledge_only=list(knowledge_only)
        )

    def generate_distribution_stats(
            self,
            train_disease_names: List[str],
            val_disease_names: List[str],
            test_disease_names: List[str],
            knowledge_disease_names: List[str],
            config: Dict[str, any]
    ) -> DatasetDistribution:
        """Generate complete distribution statistics"""

        train_diseases = set(train_disease_names)
        val_diseases = set(val_disease_names)
        test_diseases = set(test_disease_names)
        knowledge_diseases = set(knowledge_disease_names)


        train_stats = self._calculate_dataset_stats(train_disease_names)
        val_stats = self._calculate_dataset_stats(val_disease_names)
        test_stats = self._calculate_dataset_stats(test_disease_names)
        knowledge_stats = self._calculate_dataset_stats(knowledge_disease_names)


        cross_dataset_stats = self._calculate_cross_dataset_stats(
            train_diseases,
            val_diseases,
            test_diseases,
            knowledge_diseases
        )

        return DatasetDistribution(
            train_images=train_stats,
            val_images=val_stats,
            test_images=test_stats,
            knowledge=knowledge_stats,
            cross_dataset_stats=cross_dataset_stats,
            configuration=config
        )

    def save_distribution_stats(self, distribution: DatasetDistribution):
        """Save distribution statistics to JSON file"""
        try:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"data_distribution_{timestamp}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    distribution.to_dict(),
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            self.logger.info(f"Distribution statistics saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving distribution statistics: {str(e)}")
            raise