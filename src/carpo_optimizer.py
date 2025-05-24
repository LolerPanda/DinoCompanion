"""
CARPO (Child-Aware Risk-calibrated Preference Optimization) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CARPOOutput:
    """Output from CARPO optimization"""
    response: str
    preference_score: float
    risk_score: float
    risk_aware_advantage: float
    should_filter: bool
    uncertainty: float


class CARPOOptimizer(nn.Module):
    """Child-Aware Risk-calibrated Preference Optimization"""
    
    def __init__(self, config, hidden_size: int = 768):
        super().__init__()
        self.config = config
        
        # Preference and risk prediction heads
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation through dropout
        self.uncertainty_dropout = nn.Dropout(0.2)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        compute_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to compute preference and risk scores
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            compute_uncertainty: Whether to compute epistemic uncertainty
            
        Returns:
            preference_scores: Preference scores [batch_size]
            risk_scores: Risk scores [batch_size]
            uncertainty: Epistemic uncertainty [batch_size] (if computed)
        """
        # Pool hidden states (use [CLS] token or mean pooling)
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        else:
            pooled = hidden_states
        
        # Compute preference and risk scores
        preference_scores = self.preference_head(pooled).squeeze(-1)
        risk_scores = self.risk_head(pooled).squeeze(-1)
        
        # Compute uncertainty if requested
        uncertainty = None
        if compute_uncertainty:
            uncertainty = self._compute_epistemic_uncertainty(pooled)
        
        return preference_scores, risk_scores, uncertainty
    
    def compute_risk_aware_advantage(
        self,
        preference_score: float,
        risk_score: float,
        uncertainty: float
    ) -> float:
        """
        Compute risk-aware advantage: Δ(x,y) = r_p(x,y) - λ(u) * r_s(x,y)
        
        Args:
            preference_score: Preference score r_p
            risk_score: Risk score r_s
            uncertainty: Epistemic uncertainty u
            
        Returns:
            Risk-aware advantage score
        """
        # Uncertainty-adaptive weight: λ(u) = λ_0 * (1 + u)
        lambda_u = self.config.lambda_0 * (1 + uncertainty)
        
        # Risk-aware advantage
        advantage = preference_score - lambda_u * risk_score
        
        return advantage
    
    def compute_carpo_loss(
        self,
        policy_logits_chosen: torch.Tensor,
        policy_logits_rejected: torch.Tensor,
        reference_logits_chosen: torch.Tensor,
        reference_logits_rejected: torch.Tensor,
        risk_scores_chosen: torch.Tensor,
        risk_scores_rejected: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CARPO loss following Eq. 4 in the paper
        
        L_CARPO = -E[log σ(β log(π_θ(y_w)π_ref(y_l) / π_θ(y_l)π_ref(y_w)))]
                  + E[λ(u)[r_s(y_w) - r_s(y_l)]_+]
        """
        # Compute log probability ratios
        log_ratio_chosen = policy_logits_chosen - reference_logits_chosen
        log_ratio_rejected = policy_logits_rejected - reference_logits_rejected
        
        # Policy preference term
        preference_diff = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        preference_loss = -F.logsigmoid(preference_diff).mean()
        
        # Risk penalty term
        risk_diff = risk_scores_chosen - risk_scores_rejected
        risk_penalty = torch.relu(risk_diff)  # [·]_+ operator
        
        # Uncertainty-weighted risk
        lambda_u = self.config.lambda_0 * (1 + uncertainties)
        weighted_risk = (lambda_u * risk_penalty).mean()
        
        # Total CARPO loss
        total_loss = preference_loss + weighted_risk
        
        return total_loss
    
    def _compute_epistemic_uncertainty(
        self, 
        pooled_states: torch.Tensor,
        num_passes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty through multiple stochastic forward passes
        
        Args:
            pooled_states: Pooled hidden states [batch_size, hidden_size]
            num_passes: Number of stochastic passes (default from config)
            
        Returns:
            Epistemic uncertainty estimates [batch_size]
        """
        if num_passes is None:
            num_passes = self.config.uncertainty_passes
        
        # Enable dropout for uncertainty estimation
        self.train()
        
        preference_samples = []
        risk_samples = []
        
        with torch.no_grad():
            for _ in range(num_passes):
                # Apply dropout and compute scores
                dropped_states = self.uncertainty_dropout(pooled_states)
                pref = self.preference_head(dropped_states).squeeze(-1)
                risk = self.risk_head(dropped_states).squeeze(-1)
                
                preference_samples.append(pref)
                risk_samples.append(risk)
        
        # Stack samples: [num_passes, batch_size]
        preference_samples = torch.stack(preference_samples)
        risk_samples = torch.stack(risk_samples)
        
        # Compute variance as uncertainty measure
        pref_uncertainty = preference_samples.var(dim=0)
        risk_uncertainty = risk_samples.var(dim=0)
        
        # Combined uncertainty (average of both)
        uncertainty = (pref_uncertainty + risk_uncertainty) / 2
        
        return uncertainty
    
    def should_filter_response(
        self,
        risk_score: float,
        uncertainty: float,
        parent_rules: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Determine if response should be filtered based on risk
        
        Args:
            risk_score: Computed risk score
            uncertainty: Epistemic uncertainty
            parent_rules: Optional parental control rules
            
        Returns:
            should_filter: Whether to filter the response
            reason: Reason for filtering (if applicable)
        """
        # Check risk threshold
        if risk_score > self.config.risk_threshold:
            return True, f"Risk score {risk_score:.2f} exceeds threshold"
        
        # Check uncertainty-adjusted risk
        adjusted_risk = risk_score * (1 + uncertainty)
        if adjusted_risk > self.config.risk_threshold * 1.2:
            return True, f"High uncertainty risk {adjusted_risk:.2f}"
        
        # Check parent rules if provided
        if parent_rules:
            # Simplified rule checking for demo
            for rule in parent_rules:
                if "no_scary" in rule and risk_score > 0.5:
                    return True, "Parental rule: no scary content"
        
        return False, ""
    
    def generate_safe_refusal(self, child_age: int, reason: str) -> str:
        """Generate age-appropriate refusal message"""
        if child_age < 5:
            return "Oh, let's talk about something else fun instead! What's your favorite toy?"
        elif child_age < 8:
            return "Hmm, I think we should explore a different topic. What else would you like to know about?"
        else:
            return "I'd prefer to discuss something more appropriate. Is there another topic you're curious about?"


class RiskDetector:
    """Detect attachment-related risks in interactions"""
    
    def __init__(self, config):
        self.config = config
        self.risk_patterns = {
            "rejection": [
                "go away", "leave me alone", "don't want you",
                "hate you", "you're annoying"
            ],
            "inconsistency": [
                "sometimes", "maybe later", "we'll see",
                "not now", "ask someone else"
            ],
            "overwhelming": [
                "too much", "stop asking", "be quiet",
                "you're too loud", "calm down"
            ],
            "criticism": [
                "you're wrong", "that's stupid", "you can't",
                "you're not good at", "you always fail"
            ],
            "ignoring": [
                "not listening", "whatever", "don't care",
                "busy", "not important"
            ]
        }
    
    def detect_risks(
        self,
        text: str,
        emotion_state: Optional[Dict] = None,
        context_history: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Detect attachment risks in interaction
        
        Args:
            text: Input text to analyze
            emotion_state: Current emotional state
            context_history: Previous interaction context
            
        Returns:
            Risk scores for each category
        """
        text_lower = text.lower()
        risks = {}
        
        # Check for risk patterns
        for risk_type, patterns in self.risk_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 0.3
            risks[risk_type] = min(score, 1.0)
        
        # Adjust based on emotion if available
        if emotion_state:
            if emotion_state.get("primary_emotion") in ["angry", "frustrated"]:
                risks["rejection"] *= 1.2
                risks["criticism"] *= 1.2
            elif emotion_state.get("primary_emotion") == "sad":
                risks["ignoring"] *= 1.3
        
        # Consider context patterns
        if context_history:
            # Look for repeated negative patterns
            negative_count = sum(1 for ctx in context_history[-3:] 
                               if any(word in ctx.lower() 
                                     for word in ["no", "don't", "stop", "can't"]))
            if negative_count >= 2:
                for risk in risks:
                    risks[risk] *= 1.1
        
        # Normalize scores
        for risk in risks:
            risks[risk] = min(risks[risk], 1.0)
        
        return risks 