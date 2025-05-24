"""
Streamlit web application for DinoCompanion
"""
import streamlit as st
import numpy as np
from PIL import Image
import io
import soundfile as sf
from datetime import datetime
import json

# Add project to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dino_companion import DinoCompanion
from src.config import persona_config


# Page configuration
st.set_page_config(
    page_title="DinoCompanion - Child AI Assistant",
    page_icon="🦕",
    layout="wide"
)

# Initialize session state
if 'companion' not in st.session_state:
    st.session_state.companion = None
    st.session_state.messages = []
    st.session_state.initialized = False


def initialize_companion(persona, age, attachment_style, api_key=None):
    """Initialize or update companion settings"""
    st.session_state.companion = DinoCompanion(
        persona=persona,
        child_age=age,
        attachment_style=attachment_style,
        api_key=api_key
    )
    st.session_state.initialized = True


def main():
    # Header
    st.title("🦕 DinoCompanion")
    st.markdown("### An Attachment-Theory Informed AI Companion for Children")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key = st.text_input(
            "API Key (optional)",
            type="password",
            help="Enter your SiliconFlow API key for enhanced responses"
        )
        
        # Child profile
        st.subheader("Child Profile")
        child_age = st.slider("Age", 2, 10, 6)
        
        attachment_style = st.selectbox(
            "Attachment Style",
            ["secure", "anxious", "avoidant", "disorganized"],
            help="Initial assessment of child's attachment style"
        )
        
        # Persona selection
        st.subheader("Companion Persona")
        persona = st.selectbox(
            "Choose a character",
            list(persona_config.available_personas.keys())
        )
        
        # Show persona details
        if persona:
            details = persona_config.available_personas[persona]
            st.info(f"**Traits:** {', '.join(details['traits'])}")
        
        # Initialize button
        if st.button("Initialize Companion") or not st.session_state.initialized:
            initialize_companion(persona, child_age, attachment_style, api_key)
            st.success(f"✅ {persona} is ready to chat!")
    
    # Main chat interface
    if st.session_state.initialized:
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Chat with " + st.session_state.companion.persona)
            
            # Chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "child":
                        st.chat_message("user").write(msg["content"])
                    else:
                        st.chat_message("assistant").write(msg["content"])
            
            # Input area
            with st.form("chat_form", clear_on_submit=True):
                col_text, col_img, col_audio = st.columns([3, 1, 1])
                
                with col_text:
                    user_input = st.text_input(
                        "Type your message...",
                        key="user_input"
                    )
                
                with col_img:
                    uploaded_image = st.file_uploader(
                        "📷",
                        type=["jpg", "jpeg", "png"],
                        key="image_upload"
                    )
                
                with col_audio:
                    uploaded_audio = st.file_uploader(
                        "🎤",
                        type=["wav", "mp3"],
                        key="audio_upload"
                    )
                
                submitted = st.form_submit_button("Send")
            
            # Process submission
            if submitted and user_input:
                # Add user message
                st.session_state.messages.append({
                    "role": "child",
                    "content": user_input
                })
                
                # Process image if uploaded
                image = None
                if uploaded_image:
                    image = Image.open(uploaded_image)
                
                # Process audio if uploaded
                audio = None
                if uploaded_audio:
                    audio_data, _ = sf.read(io.BytesIO(uploaded_audio.read()))
                    audio = audio_data
                
                # Get response
                with st.spinner(f"{st.session_state.companion.persona} is thinking..."):
                    response = st.session_state.companion.interact(
                        text=user_input,
                        image=image,
                        audio=audio
                    )
                
                # Add companion response
                st.session_state.messages.append({
                    "role": "companion",
                    "content": response.text
                })
                
                # Rerun to update chat
                st.rerun()
        
        with col2:
            st.header("Analysis Dashboard")
            
            if st.session_state.companion.conversation_history:
                latest = st.session_state.companion.conversation_history[-1]
                latest_response = None
                
                # Find latest companion response
                for msg in reversed(st.session_state.companion.conversation_history):
                    if "companion_text" in msg:
                        latest_response = msg
                        break
                
                # Emotion detection
                st.subheader("😊 Emotion Detection")
                if "emotion" in latest:
                    emotion = latest["emotion"]
                    st.metric("Primary Emotion", emotion.capitalize())
                    
                    # Show emotion details if available
                    if "emotion_details" in latest:
                        details = latest["emotion_details"]
                        col_v, col_a = st.columns(2)
                        with col_v:
                            st.metric("Valence", f"{details.valence:.2f}")
                        with col_a:
                            st.metric("Arousal", f"{details.arousal:.2f}")
                
                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                if latest_response and "risk_score" in latest_response:
                    risk_score = latest_response["risk_score"]
                    risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                    risk_color = "🟢" if risk_score < 0.3 else "🟡" if risk_score < 0.7 else "🔴"
                    st.metric("Risk Level", f"{risk_color} {risk_level}", f"{risk_score:.2f}")
                
                # Attachment insights
                st.subheader("💝 Attachment Insights")
                profile = st.session_state.companion.attachment_profile
                st.metric("Attachment Style", profile.primary_style.capitalize())
                st.metric("Secure Base Score", 
                         f"{np.mean(list(profile.secure_base_behaviors.values())):.2f}")
                
                # Interaction quality
                quality = st.session_state.companion._assess_interaction_quality()
                st.metric("Interaction Quality", f"{quality:.2f}")
            
            # Export conversation
            st.subheader("📥 Export Data")
            if st.button("Export Conversation"):
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "settings": {
                        "persona": st.session_state.companion.persona,
                        "child_age": st.session_state.companion.child_age,
                        "attachment_style": st.session_state.companion.attachment_style
                    },
                    "messages": st.session_state.messages,
                    "analysis": {
                        "total_interactions": st.session_state.companion.interaction_count,
                        "attachment_profile": {
                            "primary_style": profile.primary_style,
                            "style_scores": profile.style_scores,
                            "secure_base_behaviors": profile.secure_base_behaviors
                        }
                    }
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"dino_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        # Welcome screen
        st.info("👈 Please configure the companion settings in the sidebar to begin.")
        
        # Show example interactions
        st.header("Example Interactions")
        
        examples = [
            {
                "persona": "Harry Potter",
                "child": "I'm scared of the dark!",
                "response": "I understand how you feel. Even wizards feel scared sometimes! Would you like to learn a special spell that helps me feel brave in the dark?"
            },
            {
                "persona": "Sun Wukong",
                "child": "I can't do this math problem.",
                "response": "Hey, even the Monkey King had to practice! Let's tackle this challenge together - what part is tricky for you?"
            },
            {
                "persona": "Dora",
                "child": "I want to explore!",
                "response": "¡Vámonos! Let's go on an adventure! Where would you like to explore today?"
            }
        ]
        
        for example in examples:
            with st.expander(f"{example['persona']} Example"):
                st.write(f"**Child:** {example['child']}")
                st.write(f"**{example['persona']}:** {example['response']}")
    
    # Footer
    st.markdown("---")
    st.caption("DinoCompanion - Supporting children's emotional development through AI")


if __name__ == "__main__":
    main() 