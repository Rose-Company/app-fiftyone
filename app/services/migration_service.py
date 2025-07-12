import json
import os
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID
from app.repositories.video import VideoRepository
from app.repositories.character import CharacterRepository
from app.repositories.character_appearance import CharacterAppearanceRepository
from app.schemas.character import CharacterCreate
from app.schemas.character_appearance import CharacterAppearanceCreate
from app.utils.exceptions import VideoNotFoundException, InvalidDataFormatException
from sqlalchemy.orm import Session

class MigrationService:
    def __init__(self, db: Session):
        self.db = db
        self.video_repo = VideoRepository(db)
        self.character_repo = CharacterRepository(db)
        self.character_appearance_repo = CharacterAppearanceRepository(db)

    def process_file(self, file_path: str) -> bool:
        try:
            # Extract video code from filename
            video_code = os.path.splitext(os.path.basename(file_path))[0]
            
            # Get video by code
            video = self.video_repo.get_by_video_code(video_code)
            if not video:
                raise VideoNotFoundException(f"Video with code {video_code} not found")

            # Read and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process each character in the file
            for character_key, appearances in data.items():
                # Extract character code and name
                character_code = character_key
                character_name = character_key.split(" (")[0]

                # Create or get character
                character_data = CharacterCreate(
                    name=character_name,
                    video_id=video.id,
                    character_code=character_code,
                    character_type="person"
                )
                
                existing_character = self.character_repo.get_by_video_id_and_code(
                    video_id=video.id,
                    character_code=character_code
                )
                
                character = existing_character or self.character_repo.create(character_data)

                # Create character appearances
                appearance_data_list = []
                for appearance in appearances:
                    start_time = Decimal(str(self._convert_time_to_seconds(appearance["start_time"])))
                    end_time = Decimal(str(self._convert_time_to_seconds(appearance["end_time"])))
                    duration = Decimal(str(appearance["duration"] / 24))  # Convert frames to seconds

                    appearance_data = CharacterAppearanceCreate(
                        character_id=character.id,
                        video_id=video.id,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        name_videoid=f"{character_name}_{video_code}"
                    )
                    appearance_data_list.append(appearance_data)

                if appearance_data_list:
                    self.character_appearance_repo.create_many(appearance_data_list)

            return True

        except Exception as e:
            # Log error and return False to indicate failure
            print(f"Error processing file {file_path}: {str(e)}")
            return False

    def _convert_time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS) to seconds"""
        h, m, s = time_str.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)

    def scan_and_process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        """
        Scan directory for new JSON files and process them
        Returns a dictionary with successful and failed files
        """
        results = {
            "success": [],
            "failed": []
        }

        # Ensure directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} not found")

        # Process each JSON file in directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                
                if self.process_file(file_path):
                    results["success"].append(filename)
                else:
                    results["failed"].append(filename)

        return results 