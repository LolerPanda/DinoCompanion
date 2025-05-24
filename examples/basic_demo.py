"""
Basic demonstration of DinoCompanion core functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dino_companion import DinoCompanion
from src.emotion_recognition import EmotionRecognizer
from src.attachment_module import AttachmentModule
from src.carpo_optimizer import RiskDetector
from src.config import attachment_config, carpo_config, system_config

def demo_emotion_recognition():
    """Demonstrate emotion recognition"""
    print("\n=== 情感识别演示 ===")
    
    recognizer = EmotionRecognizer()
    
    # Test text-based emotion recognition with both Chinese and English
    texts = [
        "我今天好开心啊！",
        "I'm so happy today!",
        "我感觉很伤心和孤独",
        "I feel sad and lonely",
        "我真的很生气！",
        "I'm really angry about this",
        "我害怕黑暗",
        "I'm scared of the dark"
    ]
    
    for text in texts:
        emotions = recognizer.recognize_from_text(text)
        print(f"\n文本: {text}")
        print("检测到的情感:")
        for emotion, score in emotions.items():
            if score > 0.1:  # Only show significant emotions
                print(f"- {emotion}: {score:.2f}")

def demo_attachment_assessment():
    """Demonstrate attachment style assessment"""
    print("\n=== 依恋风格评估演示 ===")
    
    module = AttachmentModule(attachment_config)
    
    # Create sample interaction history with Chinese
    history = [
        {"text": "我相信你", "emotion": "happy"},
        {"text": "让我们一起探索吧", "emotion": "excited"},
        {"text": "和你在一起我感觉很安全", "emotion": "calm"}
    ]
    
    behavioral_cues = {
        "seeks_comfort": 0.7,
        "independent_play": 0.6,
        "exploration_level": 0.8
    }
    
    profile = module.assess_attachment_style(history, behavioral_cues)
    
    print("\n依恋档案:")
    print(f"主要依恋风格: {profile.primary_style}")
    print("\n风格得分:")
    for style, score in profile.style_scores.items():
        print(f"- {style}: {score:.2f}")
    
    print("\n安全基地行为:")
    for behavior, score in profile.secure_base_behaviors.items():
        print(f"- {behavior}: {score:.2f}")

def demo_risk_detection():
    """Demonstrate risk detection"""
    print("\n=== 风险检测演示 ===")
    
    detector = RiskDetector(carpo_config)
    
    # Test different types of interactions in Chinese
    interactions = [
        "你做得很好！继续加油！",
        "你什么都做不好！",
        "也许我待会再帮你，看情况吧。",
        "走开，我现在很忙。"
    ]
    
    for text in interactions:
        risks = detector.detect_risks(text)
        print(f"\n文本: {text}")
        print("检测到的风险:")
        for risk_type, score in risks.items():
            if score > 0.1:  # Only show significant risks
                print(f"- {risk_type}: {score:.2f}")

def demo_basic_interaction():
    """Demonstrate basic companion interaction"""
    print("\n=== 基础交互演示 ===")
    
    # Test with different personas
    personas = ["Harry Potter", "Sun Wukong", "Dora"]
    
    for persona in personas:
        print(f"\n--- 与 {persona} 的对话 ---")
        companion = DinoCompanion(
            persona=persona,
            child_age=6,
            attachment_style="secure"
        )
        
        # Simulate a conversation in Chinese
        interactions = [
            "这个拼图太难了，我做不好！",
            "我的朋友说我不擅长拼图。",
            "你能帮我变得更好吗？",
            "和你一起玩拼图我很开心！"
        ]
        
        for text in interactions:
            print(f"\n孩子: {text}")
            response = companion.chat(text)
            print(f"{persona}: {response.text}")
            print(f"检测到的情感: {response.emotion}")
            print(f"风险等级: {response.risk_score:.2f}")
            print(f"依恋风格: {response.attachment_insights['attachment_style']}")

def demo_multilingual_support():
    """Demonstrate multilingual support"""
    print("\n=== 多语言支持演示 ===")
    
    companion = DinoCompanion(
        persona="Sun Wukong",
        child_age=6,
        attachment_style="secure"
    )
    
    # Test language switching
    print("\n测试语言切换:")
    companion.set_language("zh-CN")
    response_zh = companion.chat("你好，孙悟空！")
    print(f"中文回复: {response_zh.text}")
    
    companion.set_language("en-US")
    response_en = companion.chat("Hello, Sun Wukong!")
    print(f"English response: {response_en.text}")

def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("DINOCOMPANION 基础演示")
    print("核心功能展示")
    print("="*60)
    
    # Run demos
    demo_emotion_recognition()
    demo_attachment_assessment()
    demo_risk_detection()
    demo_basic_interaction()
    demo_multilingual_support()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60 + "\n")

if __name__ == "__main__":
    main() 