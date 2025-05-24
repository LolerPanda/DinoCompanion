"""
Basic unit tests for DinoCompanion
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dino_companion import DinoCompanion
from src.emotion_recognition import EmotionRecognizer, EmotionState
from src.attachment_module import AttachmentModule, AttachmentProfile
from src.carpo_optimizer import CARPOOptimizer, RiskDetector
from src.config import attachment_config, carpo_config


class TestDinoCompanion:
    """Test DinoCompanion main functionality"""
    
    def test_initialization(self):
        """Test companion initialization"""
        companion = DinoCompanion(
            persona="Harry Potter",
            child_age=6,
            attachment_style="secure"
        )
        
        assert companion.persona == "Harry Potter"
        assert companion.child_age == 6
        assert companion.attachment_style == "secure"
        assert companion.age_group == "5-7"
    
    def test_chat_response(self):
        """Test basic chat functionality"""
        companion = DinoCompanion()
        response = companion.chat("Hello, can you help me?")
        
        assert response.text is not None
        assert len(response.text) > 0
        assert response.emotion in ["happy", "sad", "angry", "fear", 
                                   "surprise", "disgust", "neutral"]
        assert 0 <= response.risk_score <= 1
    
    def test_persona_switching(self):
        """Test changing personas"""
        companion = DinoCompanion(persona="Harry Potter")
        
        # Switch to Sun Wukong
        companion.set_persona("Sun Wukong")
        assert companion.persona == "Sun Wukong"
        
        # Test invalid persona
        with pytest.raises(ValueError):
            companion.set_persona("Invalid Persona")
    
    def test_age_groups(self):
        """Test age group determination"""
        companion = DinoCompanion()
        
        assert companion._determine_age_group(3) == "2-4"
        assert companion._determine_age_group(6) == "5-7"
        assert companion._determine_age_group(9) == "8-10"


class TestEmotionRecognition:
    """Test emotion recognition functionality"""
    
    def test_text_emotion_recognition(self):
        """Test emotion detection from text"""
        recognizer = EmotionRecognizer()
        
        # Test happy text
        emotions = recognizer.recognize_from_text("I'm so happy and excited!")
        assert "happy" in emotions
        assert emotions["happy"] > 0.5
        
        # Test sad text
        emotions = recognizer.recognize_from_text("I feel sad and lonely")
        assert "sad" in emotions
        assert emotions["sad"] > 0.3
    
    def test_emotion_fusion(self):
        """Test multimodal emotion fusion"""
        recognizer = EmotionRecognizer()
        
        # Create mock emotion scores
        text_emotions = {"happy": 0.7, "neutral": 0.3}
        face_emotions = {"happy": 0.6, "surprise": 0.4}
        
        emotion_state = recognizer.fuse_multimodal_emotions(
            face_emotions=face_emotions,
            text_emotions=text_emotions
        )
        
        assert isinstance(emotion_state, EmotionState)
        assert emotion_state.primary_emotion == "happy"
        assert -1 <= emotion_state.valence <= 1
        assert 0 <= emotion_state.arousal <= 1


class TestAttachmentModule:
    """Test attachment theory functionality"""
    
    def test_attachment_assessment(self):
        """Test attachment style assessment"""
        module = AttachmentModule(attachment_config)
        
        # Create mock interaction history
        history = [
            {"text": "I trust you", "emotion": "happy"},
            {"text": "Let's explore together", "emotion": "excited"},
            {"text": "I feel safe with you", "emotion": "calm"}
        ]
        
        behavioral_cues = {
            "seeks_comfort": 0.7,
            "independent_play": 0.6,
            "exploration_level": 0.8
        }
        
        profile = module.assess_attachment_style(history, behavioral_cues)
        
        assert isinstance(profile, AttachmentProfile)
        assert profile.primary_style in ["secure", "anxious", "avoidant", "disorganized"]
        assert sum(profile.style_scores.values()) == pytest.approx(1.0, rel=1e-2)
    
    def test_secure_base_response(self):
        """Test secure base response generation"""
        module = AttachmentModule(attachment_config)
        
        child_state = {
            "emotion": "sad",
            "age_group": "5-7",
            "distress_level": 0.7
        }
        
        profile = AttachmentProfile(
            primary_style="anxious",
            style_scores={"anxious": 0.7, "secure": 0.3},
            secure_base_behaviors={b: 0.5 for b in attachment_config.secure_base_behaviors}
        )
        
        response = module.generate_secure_base_response(
            child_state, profile, "I'm scared"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain reassurance for anxious attachment
        assert any(phrase in response.lower() for phrase in 
                  ["here", "safe", "okay", "not going anywhere"])


class TestCARPOOptimizer:
    """Test CARPO optimization functionality"""
    
    def test_risk_detection(self):
        """Test risk detection in text"""
        detector = RiskDetector(carpo_config)
        
        # Test rejection risk
        risks = detector.detect_risks("Go away, leave me alone!")
        assert "rejection" in risks
        assert risks["rejection"] > 0.5
        
        # Test criticism risk
        risks = detector.detect_risks("You can't do anything right!")
        assert "criticism" in risks
        assert risks["criticism"] > 0.5
        
        # Test safe text
        risks = detector.detect_risks("You're doing great!")
        assert all(score < 0.3 for score in risks.values())
    
    def test_risk_aware_advantage(self):
        """Test CARPO advantage calculation"""
        optimizer = CARPOOptimizer(carpo_config)
        
        preference_score = 0.8
        risk_score = 0.3
        uncertainty = 0.1
        
        advantage = optimizer.compute_risk_aware_advantage(
            preference_score, risk_score, uncertainty
        )
        
        # Advantage should be positive for high preference, low risk
        assert advantage > 0
        
        # High risk should reduce advantage
        high_risk_advantage = optimizer.compute_risk_aware_advantage(
            preference_score, 0.9, uncertainty
        )
        assert high_risk_advantage < advantage


def test_integration():
    """Test integrated functionality"""
    companion = DinoCompanion(
        persona="Harry Potter",
        child_age=6,
        attachment_style="secure"
    )
    
    # Test a conversation flow
    responses = []
    
    # Child expresses difficulty
    response1 = companion.chat("This puzzle is too hard!")
    responses.append(response1)
    assert "help" in response1.text.lower() or "try" in response1.text.lower()
    
    # Child expresses emotion
    response2 = companion.chat("I feel frustrated")
    responses.append(response2)
    assert response2.emotion in ["frustrated", "angry", "sad"]
    
    # Check conversation history
    assert len(companion.conversation_history) >= 4  # 2 child + 2 companion
    
    # Check attachment profile evolution
    assert companion.attachment_profile.interaction_count > 0


if __name__ == "__main__":
    # Run basic tests
    print("Running DinoCompanion tests...")
    
    # Test initialization
    test_companion = TestDinoCompanion()
    test_companion.test_initialization()
    print("✓ Initialization test passed")
    
    # Test emotion recognition
    test_emotion = TestEmotionRecognition()
    test_emotion.test_text_emotion_recognition()
    print("✓ Emotion recognition test passed")
    
    # Test attachment module
    test_attachment = TestAttachmentModule()
    test_attachment.test_attachment_assessment()
    print("✓ Attachment assessment test passed")
    
    # Test CARPO
    test_carpo = TestCARPOOptimizer()
    test_carpo.test_risk_detection()
    print("✓ Risk detection test passed")
    
    # Test integration
    test_integration()
    print("✓ Integration test passed")
    
    print("\nAll tests passed! ✨") 