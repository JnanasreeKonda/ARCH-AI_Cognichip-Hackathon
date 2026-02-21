"""
Unified Agent System for Hardware Design Optimization

Hierarchical agent selection:
1. DQN (if trained checkpoint exists) - Reinforcement Learning
2. LLM (if API key available) - Gemini/GPT-4/Claude
3. Heuristic (fallback) - Rule-based search

Usage:
    from llm.unified_agent import propose_design
    params = propose_design(history)
"""

import os
import json
import random
from typing import Dict, List, Tuple

# =============================================================================
# Check Available Agents
# =============================================================================

# DQN Reinforcement Learning
DQN_AVAILABLE = False
DQN_CHECKPOINT = None
DQN_ERROR = None

try:
    import torch
    import sys
    # Check for DQN checkpoint
    possible_checkpoints = [
        'reinforcement_learning/checkpoints/dqn_final.pt',
        'reinforcement_learning/checkpoints/dqn_best.pt',
    ]
    for ckpt in possible_checkpoints:
        if os.path.exists(ckpt):
            DQN_CHECKPOINT = ckpt
            # Try to import DQN agent
            try:
                from reinforcement_learning.training.dqn_agent import DQNAgent
                DQN_AVAILABLE = True
                break
            except ImportError as e:
                DQN_ERROR = f"Failed to import DQNAgent: {e}"
                print(f"DEBUG: {DQN_ERROR}")
            except Exception as e:
                DQN_ERROR = f"Error during DQNAgent import: {e}"
                print(f"DEBUG: {DQN_ERROR}")
    
    if not DQN_CHECKPOINT:
        DQN_ERROR = "No DQN checkpoint found in reinforcement_learning/checkpoints/"

except ImportError as e:
    DQN_ERROR = f"Torch import failed: {e}"
    print(f"DEBUG: {DQN_ERROR}")

# OpenAI
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# Anthropic
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

# Gemini
GEMINI_AVAILABLE = False
try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Unified Agent Class
# =============================================================================

class UnifiedAgent:
    """
    Unified agent that automatically selects best available option:
    1. DQN (trained RL agent) if checkpoint exists
    2. LLM (Gemini/GPT-4/Claude) if API key available
    3. Heuristic (rule-based) as fallback
    """
    
    def __init__(self, mode='auto'):
        """
        Initialize unified agent with automatic selection.
        
        Args:
            mode: 'auto' (default), 'dqn', 'llm', 'heuristic'
        """
        self.mode = mode
        self.agent_type = None
        
        # Search space
        self.PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
        self.BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
        
        # Auto-detect best available agent
        if mode == 'auto':
            self._auto_select_agent()
        elif mode == 'dqn':
            if not self._init_dqn():
                self._init_heuristic()
        elif mode == 'llm':
            if not self._init_llm():
                self._init_heuristic()
        else:
            self._init_heuristic()
        
        # Ensure agent_type is set (safety fallback)
        if self.agent_type is None:
            self._init_heuristic()
        
        print(f"🤖 Agent Selected: {self.agent_type.upper()}")
    
    def _auto_select_agent(self):
        """Automatically select best available agent"""
        
        # Priority 1: DQN (if trained model exists)
        if DQN_AVAILABLE and DQN_CHECKPOINT:
            print(f"✓ Found trained DQN checkpoint: {DQN_CHECKPOINT}")
            try:
                self._init_dqn()
                return
            except Exception as e:
                print(f"⚠️  DQN initialization failed: {e}")
                print("   Falling back to LLM...")
        else:
            if DQN_ERROR:
                print(f"⚠️  DQN not available: {DQN_ERROR}")
        
        # Priority 2: LLM (if API key available)
        if self._init_llm():
            return
        
        # Priority 3: Heuristic (always available)
        self._init_heuristic()
    
    def _init_dqn(self):
        """Initialize DQN agent from checkpoint"""
        if not DQN_AVAILABLE or not DQN_CHECKPOINT:
            return False
        
        try:
            from reinforcement_learning.training.dqn_agent import DQNAgent
            # Create agent without parameters (they're loaded from checkpoint)
            self.dqn_agent = DQNAgent()
            # Load checkpoint
            self.dqn_agent.load(DQN_CHECKPOINT)
            self.dqn_agent.epsilon = 0.0  # Greedy policy (no exploration)
            self.agent_type = 'dqn'
            print(f"   Using: Trained DQN model")
            return True
        except Exception as e:
            print(f"⚠️  DQN loading failed: {e}")
            return False
    
    def _init_llm(self):
        """Initialize LLM agent (Gemini > OpenAI > Anthropic)"""
        
        # Try Gemini first
        if os.environ.get('GEMINI_API_KEY') and GEMINI_AVAILABLE:
            try:
                self.gemini_client = google_genai.Client(
                    api_key=os.environ.get('GEMINI_API_KEY')
                )
                self.agent_type = 'gemini'
                print(f"   Using: Google Gemini")
                return True
            except:
                pass
        
        # Try OpenAI
        if os.environ.get('OPENAI_API_KEY') and OPENAI_AVAILABLE:
            try:
                openai.api_key = os.environ.get('OPENAI_API_KEY')
                self.agent_type = 'openai'
                print(f"   Using: OpenAI GPT-4")
                return True
            except:
                pass
        
        # Try Anthropic
        if os.environ.get('ANTHROPIC_API_KEY') and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=os.environ.get('ANTHROPIC_API_KEY')
                )
                self.agent_type = 'anthropic'
                print(f"   Using: Anthropic Claude")
                return True
            except:
                pass
        
        return False
    
    def _init_heuristic(self):
        """Initialize heuristic agent (always works)"""
        self.agent_type = 'heuristic'
        print(f"   Using: Rule-based heuristic")
    
    # =========================================================================
    # Main Propose Design Method
    # =========================================================================
    
    def propose_design(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
        """
        Propose next design parameters based on history.
        
        Args:
            history: List of (params, metrics) tuples
            
        Returns:
            Dict with PAR and BUFFER_DEPTH parameters
        """
        if self.agent_type == 'dqn':
            return self._propose_dqn(history)
        elif self.agent_type == 'gemini':
            return self._propose_gemini(history)
        elif self.agent_type == 'openai':
            return self._propose_openai(history)
        elif self.agent_type == 'anthropic':
            return self._propose_anthropic(history)
        else:
            return self._propose_heuristic(history)
    
    # =========================================================================
    # DQN Agent
    # =========================================================================
    
    def _propose_dqn(self, history):
        """Use trained DQN agent to propose next design"""
        try:
            # Encode state
            state = self.dqn_agent.encode_state(history)
            
            # Get action from DQN (uses self.epsilon internally, already set to 0.0)
            action_raw = self.dqn_agent.select_action(state, evaluation=True)
            
            # Handle tensor/numpy/tuple/list returns - extract the actual integer
            if isinstance(action_raw, tuple):
                action = int(action_raw[0])
            elif hasattr(action_raw, 'item'):  # PyTorch tensor or numpy scalar
                action = int(action_raw.item())
            elif isinstance(action_raw, list):
                action = int(action_raw[0])
            else:
                action = int(action_raw)
            
            # Decode action to parameters manually
            # Action space: 24 actions = 6 PAR options × 4 BUFFER_DEPTH options
            par_idx = action // 4
            bd_idx = action % 4
            params = {
                'PAR': self.PAR_OPTIONS[par_idx],
                'BUFFER_DEPTH': self.BUFFER_DEPTH_OPTIONS[bd_idx]
            }
            
            print(f"💡 DQN proposed: {params}")
            return params
        except Exception as e:
            print(f"⚠️ DQN error: {e}, using heuristic fallback")
            return self._propose_heuristic(history)
    
    # =========================================================================
    # LLM Agents
    # =========================================================================
    
    def _propose_gemini(self, history):
        """Use Gemini to propose next design"""
        prompt = self._build_llm_prompt(history)
        
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            
            content = response.text.strip()
            
            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            params = json.loads(content)
            
            if self._validate_params(params):
                print(f"💡 Gemini proposed: {params}")
                return params
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
        
        return self._propose_heuristic(history)
    
    def _propose_openai(self, history):
        """Use OpenAI GPT to propose next design"""
        prompt = self._build_llm_prompt(history)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            params = json.loads(content)
            
            if self._validate_params(params):
                print(f"💡 GPT-4 proposed: {params}")
                return params
        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")
        
        return self._propose_heuristic(history)
    
    def _propose_anthropic(self, history):
        """Use Anthropic Claude to propose next design"""
        prompt = self._build_llm_prompt(history)
        
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = message.content[0].text.strip()
            params = json.loads(content)
            
            if self._validate_params(params):
                print(f"💡 Claude proposed: {params}")
                return params
        except Exception as e:
            print(f"⚠️ Anthropic error: {e}")
        
        return self._propose_heuristic(history)
    
    def _build_llm_prompt(self, history):
        """Build prompt for LLM agents"""
        history_str = self._format_history(history)
        
        return f"""
You are a hardware design optimization expert.

SEARCH SPACE:
PAR: {self.PAR_OPTIONS}
BUFFER_DEPTH: {self.BUFFER_DEPTH_OPTIONS}

OBJECTIVE:
Minimize Area-Efficiency Product (AEP):
AEP = total_cells + 0.5 * (total_cells / throughput)

HISTORY:
{history_str}

Respond ONLY with JSON:
{{"PAR": <value>, "BUFFER_DEPTH": <value>}}
"""
    
    # =========================================================================
    # Heuristic Agent
    # =========================================================================
    
    def _propose_heuristic(self, history):
        """Rule-based heuristic search fallback"""
        valid = [h for h in history if h[1].get("objective") is not None]
        
        # Initial exploration
        if len(valid) < 4:
            return random.choice([
                {"PAR": 2, "BUFFER_DEPTH": 1024},
                {"PAR": 4, "BUFFER_DEPTH": 1024},
                {"PAR": 1, "BUFFER_DEPTH": 512},
                {"PAR": 8, "BUFFER_DEPTH": 2048},
            ])
        
        # Exploit best
        best = min(valid, key=lambda x: x[1]["objective"])
        best_par = best[0]["PAR"]
        best_buffer = best[0]["BUFFER_DEPTH"]
        
        tried = {(h[0]["PAR"], h[0]["BUFFER_DEPTH"]) for h in valid}
        
        # Neighbor search
        neighbors = []
        for p in self.PAR_OPTIONS:
            for b in self.BUFFER_DEPTH_OPTIONS:
                if (p, b) not in tried:
                    neighbors.append((p, b))
        
        if neighbors:
            p, b = random.choice(neighbors)
            return {"PAR": p, "BUFFER_DEPTH": b}
        
        return {"PAR": best_par, "BUFFER_DEPTH": best_buffer}
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _format_history(self, history):
        """Format history for LLM prompt"""
        if not history:
            return "No previous designs."
        
        lines = []
        for i, (params, metrics) in enumerate(history[-10:], 1):
            lines.append(
                f"{i}. PAR={params['PAR']} "
                f"DEPTH={params['BUFFER_DEPTH']} "
                f"OBJ={metrics.get('objective', 'N/A')}"
            )
        
        return "\n".join(lines)
    
    def _validate_params(self, params):
        """Validate proposed parameters"""
        if "PAR" not in params or "BUFFER_DEPTH" not in params:
            return False
        if params["PAR"] not in self.PAR_OPTIONS:
            return False
        if params["BUFFER_DEPTH"] not in self.BUFFER_DEPTH_OPTIONS:
            return False
        return True


# =============================================================================
# Backward Compatibility
# =============================================================================

def propose_design(history):
    """
    Main entry point - automatically selects best agent.
    
    Priority:
    1. DQN (if checkpoint exists)
    2. LLM (if API key available)
    3. Heuristic (fallback)
    """
    agent = UnifiedAgent(mode='auto')
    return agent.propose_design(history)
