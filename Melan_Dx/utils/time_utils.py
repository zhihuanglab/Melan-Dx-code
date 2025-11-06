import time
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import logging
from typing import Dict, Any

@dataclass
class TimeStats:

    start_time: float
    end_time: float = 0
    duration: float = 0
    
    def stop(self):

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:

        return {
            "start_time": datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": round(self.duration, 2),
            "duration_minutes": round(self.duration / 60, 2),
            "duration_hours": round(self.duration / 3600, 2)
        }

class TimeTracker:

    def __init__(self, save_dir: str = "time_stats"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}
        self.logger = logging.getLogger(__name__)
        
    def start_track(self, name: str) -> None:

        self.stats[name] = TimeStats(start_time=time.time())
        
    def stop_track(self, name: str) -> None:

        if name in self.stats:
            self.stats[name].stop()
            
    def get_stats(self) -> Dict[str, Dict[str, Any]]:

        return {name: stat.to_dict() for name, stat in self.stats.items()}
    
    def save_stats(self) -> None:

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.save_dir / f"time_stats_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_stats(), f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Time statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving time statistics: {str(e)}")
            raise 