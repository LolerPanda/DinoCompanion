# DinoCompanion Demo
[**🔗 DINOCOMPANION Dataset on HuggingFace**](https://huggingface.co/datasets/Beaconsyh08/DINOCOMPANION)
A multimodal robot system for emotionally responsive child-AI interaction based on attachment theory and developmental psychology.

## Overview

DinoCompanion is an innovative AI system designed to provide emotionally supportive interactions with children aged 2-10. The system integrates:

- **Multimodal perception** (vision, audio, motion)
- **Attachment theory principles** for secure emotional bonding
- **Persona-driven role-play** (e.g., Harry Potter, Sun Wukong)
- **Risk-aware response generation** using CARPO optimization

## Key Features

### 1. Emotion Recognition
- Basic and complex emotion detection from facial expressions and voice
- Real-time emotional state tracking
- Adaptive response generation based on emotional context

### 2. Secure Base Behaviors
- Provides emotional comfort during distress
- Encourages exploration and learning
- Maintains consistent, predictable interaction patterns

### 3. Risk Detection
- Identifies potentially harmful caregiver interactions
- Monitors attachment risk indicators
- Implements safety boundaries in responses

### 4. Personalized Interaction
- Adapts to individual child preferences
- Maintains interaction history
- Provides developmentally appropriate responses

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dino_companion_demo.git
cd dino_companion_demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.dino_companion import DinoCompanion

# Initialize the companion
companion = DinoCompanion(
    persona="Harry Potter",
    child_age=6,
    attachment_style="secure"
)

# Process multimodal input
response = companion.interact(
    text="I'm having trouble with this puzzle",
    image=child_image,
    audio=voice_recording
)

print(response.text)
print(f"Emotion detected: {response.emotion}")
print(f"Risk level: {response.risk_score}")
```

## Architecture

The system consists of several key components:

1. **Multimodal Encoder**: Processes visual, audio, and text inputs
2. **Attachment Module**: Evaluates attachment-related behaviors
3. **CARPO Optimizer**: Balances engagement and safety
4. **Response Generator**: Creates developmentally appropriate responses

## Usage Examples

### Basic Interaction
```python
# Simple text interaction
response = companion.chat("I'm feeling sad today")
```

### Multimodal Analysis
```python
# Analyze child-caregiver interaction
analysis = companion.analyze_interaction(
    video_path="interaction.mp4",
    detect_risks=True
)
```

### Persona Customization
```python
# Switch personas
companion.set_persona("Sun Wukong")
companion.set_language("zh-CN")
```

## Evaluation

The system can be evaluated using the AttachSecure-Bench benchmark:

```bash
python -m tests.evaluate_benchmark --model dino_companion --benchmark attachsecure
```

## Safety and Ethics

DinoCompanion implements multiple safety measures:
- Content filtering for age-appropriate responses
- Risk detection for harmful interactions
- Transparent logging of all interactions
- Parental control settings

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers. 
