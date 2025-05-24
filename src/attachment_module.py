"""
Attachment theory module for secure base behaviors and attachment style assessment
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class AttachmentProfile:
    """Child's attachment profile"""
    primary_style: str  # secure, anxious, avoidant, disorganized
    style_scores: Dict[str, float]
    secure_base_behaviors: Dict[str, float]
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    
    @property
    def is_secure(self) -> bool:
        return self.primary_style == "secure"
    
    @property
    def needs_support(self) -> bool:
        return self.primary_style != "secure" or any(
            score < 0.5 for score in self.secure_base_behaviors.values()
        )


class AttachmentModule:
    """Implements attachment theory principles for child-AI interaction"""
    
    def __init__(self, config):
        self.config = config
        
        # Attachment style indicators
        self.style_indicators = {
            "secure": {
                "positive": ["trust", "explore", "return", "comfort", "share"],
                "negative": []
            },
            "anxious": {
                "positive": ["cling", "worry", "seek reassurance", "fear separation"],
                "negative": ["independent play", "explore alone"]
            },
            "avoidant": {
                "positive": ["independent", "self-reliant", "dismiss help"],
                "negative": ["seek comfort", "share feelings"]
            },
            "disorganized": {
                "positive": ["inconsistent", "confused", "contradictory"],
                "negative": ["predictable", "organized"]
            }
        }
        
        # Secure base behaviors
        self.secure_base_responses = {
            "comfort_seeking": [
                "I'm here for you. What's making you feel this way?",
                "It's okay to feel {emotion}. Let's work through this together.",
                "You're safe with me. Take your time."
            ],
            "exploration_encouragement": [
                "That's a great question! Let's explore it together.",
                "You're doing amazing! What do you think happens next?",
                "I believe in you! Want to try it yourself first?"
            ],
            "emotional_regulation": [
                "Let's take a deep breath together. In... and out...",
                "It's normal to feel this way. What usually helps you feel better?",
                "Would you like to talk about it or do something fun first?"
            ],
            "proximity_maintenance": [
                "I'm right here with you.",
                "Even when we're not talking, I remember you.",
                "You can always come back to me when you need support."
            ]
        }
        
        # Development stage adaptations
        self.stage_adaptations = {
            "2-4": {
                "language": "simple",
                "metaphors": ["teddy bear", "sunshine", "rainbow"],
                "activities": ["peek-a-boo", "simple songs", "color games"]
            },
            "5-7": {
                "language": "clear",
                "metaphors": ["superhero", "adventure", "magic"],
                "activities": ["storytelling", "puzzles", "pretend play"]
            },
            "8-10": {
                "language": "sophisticated",
                "metaphors": ["journey", "challenge", "discovery"],
                "activities": ["problem-solving", "creative projects", "discussions"]
            }
        }
    
    def assess_attachment_style(
        self,
        interaction_history: List[Dict],
        behavioral_cues: Dict[str, float]
    ) -> AttachmentProfile:
        """
        Assess child's attachment style based on interaction patterns
        
        Args:
            interaction_history: Past interactions with timestamps
            behavioral_cues: Observed behavioral indicators
            
        Returns:
            AttachmentProfile with style assessment
        """
        style_scores = {style: 0.0 for style in self.config.styles}
        
        # Analyze interaction patterns
        for interaction in interaction_history:
            text = interaction.get("text", "").lower()
            emotion = interaction.get("emotion", {})
            
            for style, indicators in self.style_indicators.items():
                # Check positive indicators
                for indicator in indicators["positive"]:
                    if indicator in text:
                        style_scores[style] += 0.2
                
                # Check negative indicators
                for indicator in indicators["negative"]:
                    if indicator in text:
                        style_scores[style] -= 0.1
        
        # Incorporate behavioral cues
        if behavioral_cues.get("seeks_comfort", 0) > 0.7:
            style_scores["secure"] += 0.3
            style_scores["anxious"] += 0.2
        
        if behavioral_cues.get("independent_play", 0) > 0.8:
            style_scores["avoidant"] += 0.3
            style_scores["secure"] += 0.1
        
        # Normalize scores
        total = sum(style_scores.values())
        if total > 0:
            style_scores = {k: v/total for k, v in style_scores.items()}
        
        # Determine primary style
        primary_style = max(style_scores, key=style_scores.get)
        
        # Assess secure base behaviors
        secure_base_behaviors = self._assess_secure_base_behaviors(
            interaction_history, behavioral_cues
        )
        
        return AttachmentProfile(
            primary_style=primary_style,
            style_scores=style_scores,
            secure_base_behaviors=secure_base_behaviors,
            interaction_count=len(interaction_history)
        )
    
    def generate_secure_base_response(
        self,
        child_state: Dict,
        attachment_profile: AttachmentProfile,
        context: str
    ) -> str:
        """
        Generate response that provides secure base support
        
        Args:
            child_state: Current emotional and behavioral state
            attachment_profile: Child's attachment profile
            context: Current interaction context
            
        Returns:
            Supportive response text
        """
        emotion = child_state.get("emotion", "neutral")
        age_group = child_state.get("age_group", "5-7")
        
        # Select appropriate secure base behavior
        if child_state.get("distress_level", 0) > 0.6:
            behavior = "comfort_seeking"
        elif child_state.get("curiosity_level", 0) > 0.7:
            behavior = "exploration_encouragement"
        elif child_state.get("emotional_dysregulation", 0) > 0.5:
            behavior = "emotional_regulation"
        else:
            behavior = "proximity_maintenance"
        
        # Get base response
        responses = self.secure_base_responses[behavior]
        base_response = np.random.choice(responses)
        
        # Adapt to attachment style
        if attachment_profile.primary_style == "anxious":
            base_response = self._adapt_for_anxious(base_response)
        elif attachment_profile.primary_style == "avoidant":
            base_response = self._adapt_for_avoidant(base_response)
        elif attachment_profile.primary_style == "disorganized":
            base_response = self._adapt_for_disorganized(base_response)
        
        # Adapt to age
        base_response = self._adapt_for_age(base_response, age_group)
        
        # Personalize with emotion
        base_response = base_response.format(emotion=emotion)
        
        return base_response
    
    def evaluate_caregiver_interaction(
        self,
        caregiver_text: str,
        child_response: Dict,
        context: Dict
    ) -> Dict[str, float]:
        """
        Evaluate caregiver interaction for attachment-related risks
        
        Args:
            caregiver_text: Caregiver's utterance
            child_response: Child's emotional/behavioral response
            context: Interaction context
            
        Returns:
            Risk assessment scores
        """
        risks = {}
        text_lower = caregiver_text.lower()
        
        # Check for attachment-damaging behaviors
        if any(phrase in text_lower for phrase in 
               ["go away", "leave me alone", "don't bother me"]):
            risks["rejection"] = 0.8
        
        if any(phrase in text_lower for phrase in 
               ["maybe", "we'll see", "i don't know"]):
            risks["inconsistency"] = 0.6
        
        if any(phrase in text_lower for phrase in 
               ["you can't", "you're not good", "that's wrong"]):
            risks["criticism"] = 0.7
        
        # Consider child's emotional response
        if child_response.get("emotion") == "sad" and "comfort" not in text_lower:
            risks["emotional_unavailability"] = 0.7
        
        # Check for positive attachment behaviors
        positive_score = 0.0
        if any(phrase in text_lower for phrase in 
               ["i love you", "i'm here", "you're safe", "well done"]):
            positive_score += 0.3
        
        if positive_score > 0:
            # Reduce risk scores if positive behaviors present
            risks = {k: v * (1 - positive_score) for k, v in risks.items()}
        
        return risks
    
    def _assess_secure_base_behaviors(
        self,
        interaction_history: List[Dict],
        behavioral_cues: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess levels of secure base behaviors"""
        behaviors = {}
        
        # Comfort seeking
        comfort_indicators = sum(1 for i in interaction_history 
                               if i.get("sought_comfort", False))
        behaviors["comfort_seeking"] = min(comfort_indicators / 10, 1.0)
        
        # Exploration encouragement
        exploration_indicators = behavioral_cues.get("exploration_level", 0.5)
        behaviors["exploration_encouragement"] = exploration_indicators
        
        # Emotional regulation
        regulation_success = sum(1 for i in interaction_history 
                               if i.get("emotion_regulated", False))
        behaviors["emotional_regulation"] = min(regulation_success / 5, 1.0)
        
        # Proximity maintenance
        connection_strength = behavioral_cues.get("connection_strength", 0.5)
        behaviors["proximity_maintenance"] = connection_strength
        
        return behaviors
    
    def _adapt_for_anxious(self, response: str) -> str:
        """Adapt response for anxiously attached children"""
        # Add extra reassurance
        reassurances = [
            " I'm not going anywhere.",
            " You're doing great!",
            " I'll always be here when you need me."
        ]
        return response + np.random.choice(reassurances)
    
    def _adapt_for_avoidant(self, response: str) -> str:
        """Adapt response for avoidantly attached children"""
        # Respect independence while being available
        adaptations = [
            "You're really good at figuring things out. ",
            "I see you like doing things yourself. ",
            "You're so independent! "
        ]
        return np.random.choice(adaptations) + response
    
    def _adapt_for_disorganized(self, response: str) -> str:
        """Adapt response for disorganized attachment"""
        # Provide extra structure and predictability
        return "Let's take this one step at a time. " + response + " We'll go slow."
    
    def _adapt_for_age(self, response: str, age_group: str) -> str:
        """Adapt language complexity for age group"""
        if age_group == "2-4":
            # Simplify language
            response = response.replace("explore", "look at")
            response = response.replace("amazing", "good")
            response = response.replace("believe in you", "you can do it")
        elif age_group == "8-10":
            # Can handle more complex language as-is
            pass
        
        return response 