"""
Configuration settings for DinoCompanion system
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel


class AttachmentConfig(BaseModel):
    """Attachment theory parameters"""
    styles: List[str] = ["secure", "anxious", "avoidant", "disorganized"]
    secure_base_behaviors: List[str] = [
        "comfort_seeking",
        "exploration_encouragement", 
        "emotional_regulation",
        "proximity_maintenance"
    ]
    risk_indicators: List[str] = [
        "rejection",
        "inconsistency",
        "overwhelming",
        "ignoring",
        "criticism"
    ]


class CARPOConfig(BaseModel):
    """CARPO optimization parameters"""
    beta: float = 0.1  # KL divergence coefficient
    lambda_0: float = 0.5  # Base risk weight
    risk_threshold: float = 0.7  # Safety threshold
    preference_weight: float = 0.6  # Preference vs risk balance
    uncertainty_passes: int = 5  # Number of stochastic passes


class ModelConfig(BaseModel):
    """Model configuration"""
    # Load model config from specified path
    config_path = "/Users/niuganjun/windsurf/siliconflow/table_caption_evaluation/llm_config.json"
    
    def __init__(self, **data):
        super().__init__(**data)
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.base_model = config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        else:
            self.base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    vision_encoder: str = "openai/clip-vit-base-patch32"
    audio_encoder: str = "facebook/wav2vec2-base"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


class PersonaConfig(BaseModel):
    """Persona configuration"""
    available_personas: Dict[str, Dict] = {
        "Harry Potter": {
            "traits": ["brave", "kind", "helpful", "magical"],
            "speech_style": "British, polite, encouraging",
            "knowledge_areas": ["magic", "friendship", "problem-solving"]
        },
        "Sun Wukong": {
            "traits": ["playful", "clever", "protective", "adventurous"],
            "speech_style": "Energetic, humorous, supportive",
            "knowledge_areas": ["adventure", "creativity", "courage"]
        },
        "Dora": {
            "traits": ["curious", "friendly", "educational", "inclusive"],
            "speech_style": "Simple, clear, interactive",
            "knowledge_areas": ["exploration", "language", "problem-solving"]
        }
    }


class DevelopmentalConfig(BaseModel):
    """Developmental psychology parameters"""
    age_groups: Dict[str, Dict] = {
        "2-4": {
            "cognitive_stage": "preoperational",
            "language_level": "simple",
            "attention_span": 5,  # minutes
            "primary_needs": ["security", "routine", "play"]
        },
        "5-7": {
            "cognitive_stage": "preoperational-concrete",
            "language_level": "moderate", 
            "attention_span": 10,
            "primary_needs": ["exploration", "social", "learning"]
        },
        "8-10": {
            "cognitive_stage": "concrete_operational",
            "language_level": "advanced",
            "attention_span": 20,
            "primary_needs": ["autonomy", "competence", "relationships"]
        }
    }
    
    theories: List[str] = [
        "attachment_theory",
        "self_efficacy",
        "theory_of_mind",
        "zone_of_proximal_development",
        "emotional_regulation",
        "social_learning",
        "cognitive_development",
        "moral_development",
        "identity_formation"
    ]


class SystemConfig(BaseModel):
    """System-wide configuration"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    model_dir: Path = project_root / "models"
    log_dir: Path = project_root / "logs"
    
    # API settings
    api_base_url: str = "https://api.siliconflow.cn/v1"
    api_key: Optional[str] = os.environ.get("SILICONFLOW_API_KEY")
    
    # Safety settings
    enable_content_filter: bool = True
    enable_risk_detection: bool = True
    log_interactions: bool = True
    
    # Performance settings
    batch_size: int = 16
    num_workers: int = 4
    cache_size: int = 1000
    
    # Language settings
    default_language: str = "zh-CN"
    supported_languages: List[str] = ["zh-CN", "en-US"]


# Global configuration instances
attachment_config = AttachmentConfig()
carpo_config = CARPOConfig()
model_config = ModelConfig()
persona_config = PersonaConfig()
developmental_config = DevelopmentalConfig()
system_config = SystemConfig()


# Ensure directories exist
for dir_path in [system_config.data_dir, system_config.model_dir, system_config.log_dir]:
    dir_path.mkdir(parents=True, exist_ok=True) 