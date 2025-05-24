"""
Simple demonstration of DinoCompanion functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dino_companion import DinoCompanion
import numpy as np
from PIL import Image


def demo_basic_interaction():
    """Demonstrate basic text interaction"""
    print("=== Basic Text Interaction Demo ===\n")
    
    # Initialize companion
    companion = DinoCompanion(
        persona="Harry Potter",
        child_age=6,
        attachment_style="secure"
    )
    
    # Simulate a conversation
    interactions = [
        "I'm having trouble with this puzzle, it's really hard!",
        "My friend said I'm not good at puzzles.",
        "Can you help me be better at solving puzzles?",
        "I feel happy when I solve puzzles with you!"
    ]
    
    for text in interactions:
        print(f"Child: {text}")
        response = companion.chat(text)
        print(f"Harry Potter: {response.text}")
        print(f"Detected emotion: {response.emotion}")
        print(f"Risk level: {response.risk_score:.2f}")
        print(f"Attachment style: {response.attachment_insights['attachment_style']}")
        print("-" * 50 + "\n")


def demo_multimodal_interaction():
    """Demonstrate multimodal interaction with simulated inputs"""
    print("=== Multimodal Interaction Demo ===\n")
    
    companion = DinoCompanion(
        persona="Sun Wukong",
        child_age=5,
        attachment_style="anxious"
    )
    
    # Simulate image (face with sad expression)
    fake_image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Simulate audio (voice recording)
    fake_audio = np.random.randn(16000)  # 1 second at 16kHz
    
    response = companion.interact(
        text="I miss my mommy",
        image=fake_image,
        audio=fake_audio
    )
    
    print(f"Child: I miss my mommy")
    print(f"Sun Wukong: {response.text}")
    print(f"\nMultimodal Analysis:")
    print(f"- Primary emotion: {response.emotion}")
    print(f"- Attachment needs support: {response.attachment_insights['needs_support']}")
    print(f"- Secure base score: {response.attachment_insights['secure_base_score']:.2f}")
    print(f"- Interaction quality: {response.attachment_insights['interaction_quality']:.2f}")


def demo_risk_detection():
    """Demonstrate risk detection in caregiver interactions"""
    print("\n=== Risk Detection Demo ===\n")
    
    companion = DinoCompanion(child_age=7)
    
    # Analyze potentially harmful caregiver responses
    caregiver_utterances = [
        ("You're doing great! Keep trying!", "Positive"),
        ("You can't do anything right!", "Critical"),
        ("Maybe I'll help you later, we'll see.", "Inconsistent"),
        ("Go away, I'm busy right now.", "Rejecting")
    ]
    
    for utterance, label in caregiver_utterances:
        # Simulate child's response to caregiver
        child_response = companion.interact(
            text=utterance,
            parent_rules=["no_criticism", "encourage_exploration"]
        )
        
        print(f"Caregiver ({label}): {utterance}")
        print(f"Risk score: {child_response.risk_score:.2f}")
        
        # Use attachment module directly for detailed analysis
        risks = companion.attachment_module.evaluate_caregiver_interaction(
            utterance,
            {"emotion": "sad"},
            {}
        )
        
        if risks:
            print("Detected risks:")
            for risk_type, score in risks.items():
                if score > 0.1:
                    print(f"  - {risk_type}: {score:.2f}")
        print("-" * 30 + "\n")


def demo_persona_switching():
    """Demonstrate switching between different personas"""
    print("=== Persona Switching Demo ===\n")
    
    companion = DinoCompanion(child_age=6)
    child_text = "Can you tell me a story?"
    
    personas = ["Harry Potter", "Sun Wukong", "Dora"]
    
    for persona in personas:
        companion.set_persona(persona)
        response = companion.chat(child_text)
        print(f"Child: {child_text}")
        print(f"{persona}: {response.text}\n")


def demo_age_adaptation():
    """Demonstrate age-appropriate responses"""
    print("=== Age Adaptation Demo ===\n")
    
    ages = [3, 6, 9]
    text = "Why is the sky blue?"
    
    for age in ages:
        companion = DinoCompanion(
            persona="Dora",
            child_age=age
        )
        
        response = companion.chat(text)
        print(f"Child (age {age}): {text}")
        print(f"Dora: {response.text}\n")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DINOCOMPANION DEMONSTRATION")
    print("Attachment-Theory Informed Child-AI Interaction")
    print("="*60 + "\n")
    
    # Run demos
    demo_basic_interaction()
    demo_multimodal_interaction()
    demo_risk_detection()
    demo_persona_switching()
    demo_age_adaptation()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main() 