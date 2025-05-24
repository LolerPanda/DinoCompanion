"""
Emotion recognition module for multimodal input processing
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EmotionState:
    """Represents detected emotional state"""
    primary_emotion: str
    confidence: float
    valence: float  # Positive/negative (-1 to 1)
    arousal: float  # Intensity (0 to 1)
    secondary_emotions: Dict[str, float]
    
    @property
    def is_complex(self) -> bool:
        """Check if emotion is complex (multiple emotions detected)"""
        return len([v for v in self.secondary_emotions.values() if v > 0.3]) > 1


class EmotionRecognizer:
    """Multimodal emotion recognition system"""
    
    # Basic emotions (Ekman's universal emotions)
    BASIC_EMOTIONS = [
        "happy", "sad", "angry", "fear", 
        "surprise", "disgust", "neutral"
    ]
    
    # Complex emotions
    COMPLEX_EMOTIONS = [
        "anxious", "excited", "frustrated", "confused",
        "proud", "ashamed", "jealous", "grateful"
    ]
    
    def __init__(self, model_config=None):
        """Initialize emotion recognition models"""
        self.model_config = model_config
        self.device = torch.device(model_config.device if model_config else "cpu")
        
        # Placeholder for actual models
        self.face_model = None
        self.voice_model = None
        self.text_model = None
        
        # Emotion mapping for children
        self.child_emotion_map = {
            "happy": {"valence": 0.8, "arousal": 0.6},
            "sad": {"valence": -0.7, "arousal": 0.3},
            "angry": {"valence": -0.8, "arousal": 0.8},
            "fear": {"valence": -0.6, "arousal": 0.7},
            "surprise": {"valence": 0.1, "arousal": 0.8},
            "disgust": {"valence": -0.5, "arousal": 0.5},
            "neutral": {"valence": 0.0, "arousal": 0.3}
        }
    
    def recognize_from_face(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """Recognize emotions from facial expression"""
        # Convert image if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Preprocess image
        face = self._preprocess_face(image)
        
        # Simulate emotion detection (replace with actual model)
        emotions = self._simulate_face_emotions(face)
        
        return emotions
    
    def recognize_from_voice(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Recognize emotions from voice prosody"""
        # Extract audio features
        features = self._extract_audio_features(audio, sample_rate)
        
        # Simulate emotion detection
        emotions = self._simulate_voice_emotions(features)
        
        return emotions
    
    def recognize_from_text(self, text: str) -> Dict[str, float]:
        """Recognize emotions from text content"""
        # Simple keyword-based emotion detection for demo
        emotions = self._analyze_text_emotions(text)
        
        return emotions
    
    def fuse_multimodal_emotions(
        self, 
        face_emotions: Optional[Dict[str, float]] = None,
        voice_emotions: Optional[Dict[str, float]] = None,
        text_emotions: Optional[Dict[str, float]] = None,
        weights: Dict[str, float] = None
    ) -> EmotionState:
        """Fuse emotions from multiple modalities"""
        if weights is None:
            weights = {"face": 0.5, "voice": 0.3, "text": 0.2}
        
        # Combine emotion scores
        all_emotions = {}
        
        if face_emotions:
            for emotion, score in face_emotions.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + score * weights["face"]
        
        if voice_emotions:
            for emotion, score in voice_emotions.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + score * weights["voice"]
        
        if text_emotions:
            for emotion, score in text_emotions.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + score * weights["text"]
        
        # Normalize scores
        total = sum(all_emotions.values())
        if total > 0:
            all_emotions = {k: v/total for k, v in all_emotions.items()}
        
        # Determine primary emotion
        primary_emotion = max(all_emotions, key=all_emotions.get)
        confidence = all_emotions[primary_emotion]
        
        # Calculate valence and arousal
        valence, arousal = self._calculate_valence_arousal(all_emotions)
        
        # Get secondary emotions
        secondary_emotions = {k: v for k, v in all_emotions.items() 
                            if k != primary_emotion and v > 0.1}
        
        return EmotionState(
            primary_emotion=primary_emotion,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            secondary_emotions=secondary_emotions
        )
    
    def _preprocess_face(self, image: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion recognition"""
        # Resize to standard size
        face = cv2.resize(image, (224, 224))
        
        # Convert to grayscale if needed
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # Normalize
        face = face.astype(np.float32) / 255.0
        
        return face
    
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract prosodic features from audio"""
        # Simple feature extraction (placeholder)
        # In real implementation, extract pitch, energy, MFCCs, etc.
        features = np.array([
            np.mean(audio),  # Mean amplitude
            np.std(audio),   # Amplitude variation
            len(audio) / sample_rate,  # Duration
        ])
        
        return features
    
    def _simulate_face_emotions(self, face: np.ndarray) -> Dict[str, float]:
        """Simulate facial emotion detection"""
        # Random scores for demo (replace with actual model)
        emotions = {}
        for emotion in self.BASIC_EMOTIONS:
            emotions[emotion] = np.random.random() * 0.5
        
        # Ensure one dominant emotion
        dominant = np.random.choice(self.BASIC_EMOTIONS)
        emotions[dominant] += 0.5
        
        # Normalize
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _simulate_voice_emotions(self, features: np.ndarray) -> Dict[str, float]:
        """Simulate voice emotion detection"""
        # Simple rules based on features
        emotions = {emotion: 0.1 for emotion in self.BASIC_EMOTIONS}
        
        # High energy might indicate excitement or anger
        if features[1] > 0.5:  # High variation
            emotions["excited"] = 0.4
            emotions["angry"] = 0.3
        else:
            emotions["sad"] = 0.3
            emotions["neutral"] = 0.4
        
        return emotions
    
    def _analyze_text_emotions(self, text: str) -> Dict[str, float]:
        """Simple keyword-based emotion analysis"""
        text_lower = text.lower()
        
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "fun", "great", "love"],
            "sad": ["sad", "cry", "miss", "lonely", "hurt"],
            "angry": ["angry", "mad", "hate", "stupid", "mean"],
            "fear": ["scared", "afraid", "worry", "nightmare"],
            "confused": ["confused", "don't understand", "hard", "difficult"]
        }
        
        emotions = {emotion: 0.0 for emotion in self.BASIC_EMOTIONS}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotions[emotion] += 0.3
        
        # Default to neutral if no keywords found
        if sum(emotions.values()) == 0:
            emotions["neutral"] = 1.0
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def _calculate_valence_arousal(self, emotions: Dict[str, float]) -> Tuple[float, float]:
        """Calculate valence and arousal from emotion distribution"""
        valence = 0.0
        arousal = 0.0
        
        for emotion, score in emotions.items():
            if emotion in self.child_emotion_map:
                valence += score * self.child_emotion_map[emotion]["valence"]
                arousal += score * self.child_emotion_map[emotion]["arousal"]
        
        return valence, arousal
    
    def adapt_to_child_age(self, emotion_state: EmotionState, child_age: int) -> EmotionState:
        """Adapt emotion interpretation based on child's age"""
        # Younger children may express emotions differently
        if child_age < 5:
            # Simplify complex emotions to basic ones
            if emotion_state.primary_emotion in self.COMPLEX_EMOTIONS:
                # Map to nearest basic emotion
                if emotion_state.valence > 0:
                    emotion_state.primary_emotion = "happy"
                else:
                    emotion_state.primary_emotion = "sad" if emotion_state.arousal < 0.5 else "angry"
        
        return emotion_state 