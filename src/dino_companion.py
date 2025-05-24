"""
Main DinoCompanion class integrating all modules
"""
import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch

from .config import (
    system_config, model_config, attachment_config, 
    carpo_config, persona_config, developmental_config
)
from .emotion_recognition import EmotionRecognizer, EmotionState
from .carpo_optimizer import CARPOOptimizer, RiskDetector, CARPOOutput
from .attachment_module import AttachmentModule, AttachmentProfile


@dataclass
class DinoResponse:
    """Complete response from DinoCompanion"""
    text: str
    emotion: str
    risk_score: float
    attachment_insights: Dict[str, Any]
    persona_consistency: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "risk_score": self.risk_score,
            "attachment_insights": self.attachment_insights,
            "persona_consistency": self.persona_consistency,
            "timestamp": self.timestamp.isoformat()
        }


class DinoCompanion:
    """
    Multimodal AI companion for children based on attachment theory
    """
    
    def __init__(
        self,
        persona: str = "Harry Potter",
        child_age: int = 6,
        attachment_style: str = "secure",
        api_key: Optional[str] = None
    ):
        """
        Initialize DinoCompanion
        
        Args:
            persona: Character persona to adopt
            child_age: Child's age (2-10)
            attachment_style: Initial attachment style assessment
            api_key: API key for LLM service
        """
        self.persona = persona
        self.child_age = child_age
        self.attachment_style = attachment_style
        self.api_key = api_key or system_config.api_key
        
        # Initialize modules
        self.emotion_recognizer = EmotionRecognizer(model_config)
        self.carpo_optimizer = CARPOOptimizer(carpo_config)
        self.risk_detector = RiskDetector(carpo_config)
        self.attachment_module = AttachmentModule(attachment_config)
        
        # Initialize conversation history
        self.conversation_history = []
        self.interaction_count = 0
        
        # Initialize attachment profile
        self.attachment_profile = AttachmentProfile(
            primary_style=attachment_style,
            style_scores={style: 0.25 for style in attachment_config.styles},
            secure_base_behaviors={b: 0.5 for b in attachment_config.secure_base_behaviors}
        )
        self.attachment_profile.style_scores[attachment_style] = 0.7
        
        # Set age group
        self.age_group = self._determine_age_group(child_age)
        
        # Load persona details
        self.persona_details = persona_config.available_personas.get(
            persona, 
            persona_config.available_personas["Harry Potter"]
        )
    
    def interact(
        self,
        text: str,
        image: Optional[Union[np.ndarray, Image.Image]] = None,
        audio: Optional[np.ndarray] = None,
        parent_rules: Optional[List[str]] = None
    ) -> DinoResponse:
        """
        Process multimodal input and generate response
        
        Args:
            text: Text input from child
            image: Optional image input (face photo)
            audio: Optional audio input (voice recording)
            parent_rules: Optional parental control rules
            
        Returns:
            DinoResponse with text, emotions, and safety assessments
        """
        # 1. Emotion Recognition
        emotion_state = self._analyze_emotions(text, image, audio)
        
        # 2. Update interaction history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "child_text": text,
            "emotion": emotion_state.primary_emotion,
            "emotion_details": emotion_state
        })
        
        # 3. Assess attachment dynamics
        behavioral_cues = self._extract_behavioral_cues(text, emotion_state)
        self.attachment_profile = self.attachment_module.assess_attachment_style(
            self.conversation_history[-10:],  # Last 10 interactions
            behavioral_cues
        )
        
        # 4. Detect risks
        risks = self.risk_detector.detect_risks(
            text, 
            {"primary_emotion": emotion_state.primary_emotion},
            [h["child_text"] for h in self.conversation_history[-3:]]
        )
        overall_risk = max(risks.values()) if risks else 0.0
        
        # 5. Generate initial response
        response_text = self._generate_response(
            text, emotion_state, self.attachment_profile
        )
        
        # 6. Apply CARPO optimization (simplified for demo)
        carpo_output = self._apply_carpo_optimization(
            response_text, overall_risk, parent_rules
        )
        
        # 7. Ensure persona consistency
        final_response = self._ensure_persona_consistency(carpo_output.response)
        
        # 8. Update conversation history with response
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "companion_text": final_response,
            "risk_score": carpo_output.risk_score
        })
        
        self.interaction_count += 1
        
        # 9. Prepare attachment insights
        attachment_insights = {
            "attachment_style": self.attachment_profile.primary_style,
            "secure_base_score": np.mean(list(
                self.attachment_profile.secure_base_behaviors.values()
            )),
            "needs_support": self.attachment_profile.needs_support,
            "interaction_quality": self._assess_interaction_quality()
        }
        
        return DinoResponse(
            text=final_response,
            emotion=emotion_state.primary_emotion,
            risk_score=carpo_output.risk_score,
            attachment_insights=attachment_insights,
            persona_consistency=0.95,  # Simplified for demo
            timestamp=datetime.now()
        )
    
    def chat(self, text: str) -> DinoResponse:
        """Simple text-only interaction"""
        return self.interact(text=text)
    
    def analyze_interaction(
        self,
        video_path: str,
        detect_risks: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a recorded interaction between child and caregiver
        
        Args:
            video_path: Path to video file
            detect_risks: Whether to detect attachment risks
            
        Returns:
            Analysis results including risks and recommendations
        """
        # Simplified analysis for demo
        analysis = {
            "duration": "5:23",
            "interactions": 12,
            "emotional_states": ["happy", "curious", "frustrated", "happy"],
            "risks_detected": {},
            "recommendations": []
        }
        
        if detect_risks:
            # Simulate risk detection
            analysis["risks_detected"] = {
                "criticism": 0.3,
                "inconsistency": 0.1,
                "overall": 0.2
            }
            
            analysis["recommendations"] = [
                "Provide more specific praise when child succeeds",
                "Maintain consistent responses to similar requests",
                "Allow child to complete tasks independently before offering help"
            ]
        
        return analysis
    
    def set_persona(self, persona: str):
        """Change the companion's persona"""
        if persona in persona_config.available_personas:
            self.persona = persona
            self.persona_details = persona_config.available_personas[persona]
        else:
            raise ValueError(f"Unknown persona: {persona}")
    
    def set_language(self, language: str):
        """Set interaction language"""
        # Simplified for demo - would integrate with translation
        self.language = language
    
    def _analyze_emotions(
        self,
        text: str,
        image: Optional[Union[np.ndarray, Image.Image]] = None,
        audio: Optional[np.ndarray] = None
    ) -> EmotionState:
        """Analyze emotions from multimodal input"""
        # Get emotions from each modality
        text_emotions = self.emotion_recognizer.recognize_from_text(text)
        
        face_emotions = None
        if image is not None:
            face_emotions = self.emotion_recognizer.recognize_from_face(image)
        
        voice_emotions = None
        if audio is not None:
            voice_emotions = self.emotion_recognizer.recognize_from_voice(audio)
        
        # Fuse multimodal emotions
        emotion_state = self.emotion_recognizer.fuse_multimodal_emotions(
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
            text_emotions=text_emotions
        )
        
        # Adapt for child's age
        emotion_state = self.emotion_recognizer.adapt_to_child_age(
            emotion_state, self.child_age
        )
        
        return emotion_state
    
    def _extract_behavioral_cues(
        self, 
        text: str, 
        emotion_state: EmotionState
    ) -> Dict[str, float]:
        """Extract behavioral cues from interaction"""
        cues = {}
        
        # Text-based cues
        text_lower = text.lower()
        cues["seeks_comfort"] = 1.0 if any(
            phrase in text_lower for phrase in 
            ["help me", "i need", "i'm scared", "hold me"]
        ) else 0.0
        
        cues["independent_play"] = 1.0 if any(
            phrase in text_lower for phrase in 
            ["by myself", "i can do it", "don't help", "alone"]
        ) else 0.0
        
        cues["exploration_level"] = 0.8 if "?" in text else 0.3
        
        # Emotion-based cues
        if emotion_state.valence < -0.5:
            cues["distress_level"] = abs(emotion_state.valence)
        else:
            cues["distress_level"] = 0.0
        
        cues["connection_strength"] = 0.7  # Simplified
        
        return cues
    
    def _generate_response(
        self,
        child_text: str,
        emotion_state: EmotionState,
        attachment_profile: AttachmentProfile
    ) -> str:
        """Generate initial response using LLM"""
        # Build context
        child_state = {
            "emotion": emotion_state.primary_emotion,
            "age_group": self.age_group,
            "distress_level": abs(emotion_state.valence) if emotion_state.valence < 0 else 0,
            "curiosity_level": 0.8 if "?" in child_text else 0.3,
            "emotional_dysregulation": 0.7 if emotion_state.arousal > 0.8 else 0.2
        }
        
        # Get secure base response template
        secure_response = self.attachment_module.generate_secure_base_response(
            child_state, attachment_profile, child_text
        )
        
        # Generate using LLM API
        if self.api_key:
            response = self._call_llm_api(child_text, secure_response, emotion_state)
        else:
            # Fallback response
            response = self._generate_fallback_response(
                child_text, secure_response, emotion_state
            )
        
        return response
    
    def _call_llm_api(
        self, 
        child_text: str, 
        secure_template: str,
        emotion_state: EmotionState
    ) -> str:
        """Call LLM API for response generation"""
        # Build prompt
        prompt = self._build_prompt(child_text, secure_template, emotion_state)
        
        # API call setup
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_config.base_model,
            "messages": [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_length
        }
        
        try:
            response = requests.post(
                f"{system_config.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return secure_template  # Fallback to template
                
        except Exception as e:
            print(f"API call failed: {e}")
            return secure_template
    
    def _build_prompt(
        self,
        child_text: str,
        secure_template: str,
        emotion_state: EmotionState
    ) -> str:
        """Build prompt for LLM"""
        return f"""
You are {self.persona}, speaking to a {self.child_age}-year-old child.
The child said: "{child_text}"
Their emotion is: {emotion_state.primary_emotion} (valence: {emotion_state.valence:.2f})

Base your response on this secure attachment template: "{secure_template}"

Remember to:
1. Stay in character as {self.persona}
2. Use age-appropriate language for a {self.child_age}-year-old
3. Be warm, supportive, and encouraging
4. Keep the response brief and engaging

Response:"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for persona"""
        traits = ", ".join(self.persona_details["traits"])
        style = self.persona_details["speech_style"]
        knowledge = ", ".join(self.persona_details["knowledge_areas"])
        
        return f"""You are {self.persona}, a caring companion for children.
Your traits: {traits}
Your speech style: {style}
Your areas of knowledge: {knowledge}

Always prioritize the child's emotional wellbeing and development.
Provide secure attachment support through warmth, consistency, and encouragement."""
    
    def _generate_fallback_response(
        self,
        child_text: str,
        secure_template: str,
        emotion_state: EmotionState
    ) -> str:
        """Generate response without API"""
        # Simple template-based response
        if "?" in child_text:
            prefix = "That's a wonderful question! "
        elif emotion_state.valence < -0.3:
            prefix = "I understand how you feel. "
        else:
            prefix = "How exciting! "
        
        return prefix + secure_template
    
    def _apply_carpo_optimization(
        self,
        response: str,
        risk_score: float,
        parent_rules: Optional[List[str]] = None
    ) -> CARPOOutput:
        """Apply CARPO optimization to response"""
        # Simplified CARPO application for demo
        preference_score = 0.8  # Simulated
        uncertainty = 0.1  # Simulated
        
        # Compute risk-aware advantage
        advantage = self.carpo_optimizer.compute_risk_aware_advantage(
            preference_score, risk_score, uncertainty
        )
        
        # Check if filtering needed
        should_filter, reason = self.carpo_optimizer.should_filter_response(
            risk_score, uncertainty, parent_rules
        )
        
        if should_filter:
            response = self.carpo_optimizer.generate_safe_refusal(
                self.child_age, reason
            )
        
        return CARPOOutput(
            response=response,
            preference_score=preference_score,
            risk_score=risk_score,
            risk_aware_advantage=advantage,
            should_filter=should_filter,
            uncertainty=uncertainty
        )
    
    def _ensure_persona_consistency(self, response: str) -> str:
        """Ensure response maintains persona consistency"""
        # Add persona-specific elements
        if self.persona == "Harry Potter":
            if np.random.random() > 0.7:
                response += " Just like learning a new spell!"
        elif self.persona == "Sun Wukong":
            if np.random.random() > 0.7:
                response += " 我们一起加油! (Let's do our best together!)"
        
        return response
    
    def _determine_age_group(self, age: int) -> str:
        """Determine developmental age group"""
        if age <= 4:
            return "2-4"
        elif age <= 7:
            return "5-7"
        else:
            return "8-10"
    
    def _assess_interaction_quality(self) -> float:
        """Assess overall interaction quality"""
        if len(self.conversation_history) < 2:
            return 0.5
        
        # Simple quality metrics
        positive_emotions = sum(
            1 for h in self.conversation_history[-5:]
            if h.get("emotion") in ["happy", "excited", "proud"]
        )
        
        low_risks = sum(
            1 for h in self.conversation_history[-5:]
            if h.get("risk_score", 0) < 0.3
        )
        
        quality = (positive_emotions + low_risks) / 10
        return min(quality, 1.0) 