# """
# LLM-Powered Design Space Exploration Agent

# Supports:
# - OpenAI (GPT-4, GPT-3.5)
# - Anthropic (Claude)
# - Fallback to heuristic search if no API key
# """

# import os
# import json
# import random
# from typing import Dict, List, Tuple, Any

# # Try importing LLM libraries (optional)
# try:
#     import openai
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# try:
#     import anthropic
#     ANTHROPIC_AVAILABLE = True
# except ImportError:
#     ANTHROPIC_AVAILABLE = False


# class DesignAgent:
#     """LLM-powered or heuristic design space exploration agent."""
    
#     def __init__(self, mode='auto'):
#         """
#         Initialize agent.
        
#         Args:
#             mode: 'openai', 'anthropic', 'heuristic', or 'auto' (auto-detect)
#         """
#         self.mode = mode
        
#         # Search space definition
#         self.PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
#         self.BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
        
#         # Auto-detect available LLM
#         if mode == 'auto':
#             if os.environ.get('OPENAI_API_KEY') and OPENAI_AVAILABLE:
#                 self.mode = 'openai'
#                 openai.api_key = os.environ.get('OPENAI_API_KEY')
#             elif os.environ.get('ANTHROPIC_API_KEY') and ANTHROPIC_AVAILABLE:
#                 self.mode = 'anthropic'
#                 self.anthropic_client = anthropic.Anthropic(
#                     api_key=os.environ.get('ANTHROPIC_API_KEY')
#                 )
#             else:
#                 self.mode = 'heuristic'
        
#         print(f"🤖 Agent Mode: {self.mode.upper()}")
    
#     def propose_design(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
#         """
#         Propose next design parameters based on history.
        
#         Args:
#             history: List of (params, metrics) tuples
            
#         Returns:
#             Dict with PAR and BUFFER_DEPTH parameters
#         """
#         if self.mode == 'openai':
#             return self._propose_openai(history)
#         elif self.mode == 'anthropic':
#             return self._propose_anthropic(history)
#         else:
#             return self._propose_heuristic(history)
    
#     def _propose_openai(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
#         """Use OpenAI GPT to propose next design."""
        
#         # Format history for LLM
#         history_str = self._format_history(history)
        
#         prompt = f"""You are a hardware design optimization expert. Your task is to propose the next microarchitecture parameters to explore.

# SEARCH SPACE:
# - PAR (parallelism): {self.PAR_OPTIONS}
# - BUFFER_DEPTH: {self.BUFFER_DEPTH_OPTIONS}

# OBJECTIVE: Minimize Area-Efficiency Product (AEP)
# - AEP = total_cells + 0.5 * (total_cells / throughput)
# - Lower is better

# EXPLORATION HISTORY:
# {history_str}

# INSTRUCTIONS:
# 1. Analyze the pattern in the data
# 2. Identify promising unexplored regions
# 3. Balance exploration (trying new areas) vs exploitation (refining good designs)
# 4. Propose the next PAR and BUFFER_DEPTH to try

# Respond with ONLY a JSON object in this exact format:
# {{"PAR": <value>, "BUFFER_DEPTH": <value>}}

# Do not include any explanation, just the JSON."""

#         try:
#             response = openai.ChatCompletion.create(
#                 model="gpt-4",
#                 messages=[
#                     {"role": "system", "content": "You are a hardware optimization expert. Always respond with valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.7,
#                 max_tokens=100
#             )
            
#             # Parse response
#             content = response.choices[0].message.content.strip()
#             params = json.loads(content)
            
#             # Validate
#             if self._validate_params(params):
#                 print(f"💡 LLM proposed: PAR={params['PAR']}, BUFFER_DEPTH={params['BUFFER_DEPTH']}")
#                 return params
#             else:
#                 print("⚠️  LLM proposed invalid params, using heuristic fallback")
#                 return self._propose_heuristic(history)
                
#         except Exception as e:
#             print(f"⚠️  OpenAI API error: {e}, using heuristic fallback")
#             return self._propose_heuristic(history)
    
#     def _propose_anthropic(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
#         """Use Anthropic Claude to propose next design."""
        
#         history_str = self._format_history(history)
        
#         prompt = f"""You are a hardware design optimization expert. Your task is to propose the next microarchitecture parameters to explore.

# SEARCH SPACE:
# - PAR (parallelism): {self.PAR_OPTIONS}
# - BUFFER_DEPTH: {self.BUFFER_DEPTH_OPTIONS}

# OBJECTIVE: Minimize Area-Efficiency Product (AEP)
# - AEP = total_cells + 0.5 * (total_cells / throughput)
# - Lower is better

# EXPLORATION HISTORY:
# {history_str}

# INSTRUCTIONS:
# 1. Analyze the pattern in the data
# 2. Identify promising unexplored regions
# 3. Balance exploration (trying new areas) vs exploitation (refining good designs)
# 4. Propose the next PAR and BUFFER_DEPTH to try

# Respond with ONLY a JSON object in this exact format:
# {{"PAR": <value>, "BUFFER_DEPTH": <value>}}

# Do not include any explanation, just the JSON."""

#         try:
#             message = self.anthropic_client.messages.create(
#                 model="claude-3-5-sonnet-20241022",
#                 max_tokens=100,
#                 temperature=0.7,
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ]
#             )
            
#             # Parse response
#             content = message.content[0].text.strip()
#             params = json.loads(content)
            
#             # Validate
#             if self._validate_params(params):
#                 print(f"💡 LLM proposed: PAR={params['PAR']}, BUFFER_DEPTH={params['BUFFER_DEPTH']}")
#                 return params
#             else:
#                 print("⚠️  LLM proposed invalid params, using heuristic fallback")
#                 return self._propose_heuristic(history)
                
#         except Exception as e:
#             print(f"⚠️  Anthropic API error: {e}, using heuristic fallback")
#             return self._propose_heuristic(history)
    
#     def _propose_heuristic(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
#         """Heuristic search fallback (same as before)."""
        
#         valid = [h for h in history if h[1].get("total_cells") is not None]
        
#         # Phase 1: Initial exploration
#         if len(valid) < 4:
#             if len(valid) == 0:
#                 return {"PAR": 2, "BUFFER_DEPTH": 1024}
#             elif len(valid) == 1:
#                 return {"PAR": 4, "BUFFER_DEPTH": 1024}
#             elif len(valid) == 2:
#                 return {"PAR": 1, "BUFFER_DEPTH": 512}
#             else:
#                 return {"PAR": 8, "BUFFER_DEPTH": 2048}
        
#         # Phase 2: Exploitation around best
#         best = min(valid, key=lambda x: x[1].get("objective", float('inf')))
#         best_params = best[0]
#         best_par = best_params["PAR"]
#         best_buffer = best_params.get("BUFFER_DEPTH", 1024)
        
#         try:
#             par_idx = self.PAR_OPTIONS.index(best_par)
#             buffer_idx = self.BUFFER_DEPTH_OPTIONS.index(best_buffer)
#         except ValueError:
#             par_idx = 2
#             buffer_idx = 2
        
#         # 70% exploitation, 30% exploration
#         if random.random() < 0.7:
#             neighbor_pars = []
#             if par_idx > 0:
#                 neighbor_pars.append(self.PAR_OPTIONS[par_idx - 1])
#             if par_idx < len(self.PAR_OPTIONS) - 1:
#                 neighbor_pars.append(self.PAR_OPTIONS[par_idx + 1])
#             neighbor_pars.append(best_par)
            
#             neighbor_buffers = []
#             if buffer_idx > 0:
#                 neighbor_buffers.append(self.BUFFER_DEPTH_OPTIONS[buffer_idx - 1])
#             if buffer_idx < len(self.BUFFER_DEPTH_OPTIONS) - 1:
#                 neighbor_buffers.append(self.BUFFER_DEPTH_OPTIONS[buffer_idx + 1])
#             neighbor_buffers.append(best_buffer)
            
#             neighbors = [(p, b) for p in neighbor_pars for b in neighbor_buffers]
#             tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in valid}
#             untried_neighbors = [(p, b) for p, b in neighbors if (p, b) not in tried]
            
#             if untried_neighbors:
#                 par, buffer = random.choice(untried_neighbors)
#                 return {"PAR": par, "BUFFER_DEPTH": buffer}
        
#         # Random exploration
#         tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in valid}
#         untried = [(p, b) for p in self.PAR_OPTIONS for b in self.BUFFER_DEPTH_OPTIONS 
#                    if (p, b) not in tried]
        
#         if untried:
#             par, buffer = random.choice(untried)
#             return {"PAR": par, "BUFFER_DEPTH": buffer}
        
#         return {"PAR": best_par, "BUFFER_DEPTH": best_buffer}
    
#     def _format_history(self, history: List[Tuple[Dict, Dict]]) -> str:
#         """Format history for LLM prompt."""
#         if not history:
#             return "No previous designs yet."
        
#         lines = ["Iteration | PAR | DEPTH | Cells | FFs | Logic | Objective"]
#         lines.append("-" * 65)
        
#         for i, (params, metrics) in enumerate(history[-10:], 1):  # Last 10 only
#             par = params['PAR']
#             depth = params.get('BUFFER_DEPTH', 1024)
#             cells = metrics.get('total_cells', 'N/A')
#             ffs = metrics.get('flip_flops', 'N/A')
#             logic = metrics.get('logic_cells', 'N/A')
#             obj = metrics.get('objective', 'N/A')
            
#             if isinstance(obj, float):
#                 obj = f"{obj:.1f}"
            
#             lines.append(f"{i:4d}      | {par:3d} | {depth:5d} | {cells:5} | {ffs:4} | {logic:5} | {obj:8}")
        
#         # Add best design info
#         if history:
#             best = min(history, key=lambda x: x[1].get('objective', float('inf')))
#             best_params, best_metrics = best
#             lines.append("\nBEST DESIGN SO FAR:")
#             lines.append(f"  PAR={best_params['PAR']}, DEPTH={best_params.get('BUFFER_DEPTH', 1024)}")
#             lines.append(f"  Objective={best_metrics.get('objective', 'N/A'):.1f}")
        
#         return "\n".join(lines)
    
#     def _validate_params(self, params: Dict) -> bool:
#         """Validate proposed parameters."""
#         if 'PAR' not in params or 'BUFFER_DEPTH' not in params:
#             return False
#         if params['PAR'] not in self.PAR_OPTIONS:
#             return False
#         if params['BUFFER_DEPTH'] not in self.BUFFER_DEPTH_OPTIONS:
#             return False
#         return True


# # Backward compatibility function
# def propose_design(history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:
#     """
#     Main entry point for backward compatibility.
#     Auto-detects and uses best available agent.
#     """
#     agent = DesignAgent(mode='auto')
#     return agent.propose_design(history)


import os
import json
import random
from typing import List, Tuple, Dict

# -----------------------------
# LLM Availability Detection
# -----------------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class DesignAgent:
    """LLM-powered or heuristic design space exploration agent."""

    def __init__(self, mode='auto'):
        """
        mode: 'openai', 'anthropic', 'gemini', 'heuristic', or 'auto'
        """

        self.mode = mode

        # Search space
        self.PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
        self.BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]

        # -----------------------------
        # Auto-detect best available LLM
        # -----------------------------
        if mode == 'auto':
            if os.environ.get('OPENAI_API_KEY') and OPENAI_AVAILABLE:
                self.mode = 'openai'
                openai.api_key = os.environ.get('OPENAI_API_KEY')

            elif os.environ.get('ANTHROPIC_API_KEY') and ANTHROPIC_AVAILABLE:
                self.mode = 'anthropic'
                self.anthropic_client = anthropic.Anthropic(
                    api_key=os.environ.get('ANTHROPIC_API_KEY')
                )

            elif os.environ.get('GEMINI_API_KEY') and GEMINI_AVAILABLE:
                self.mode = 'gemini'
                self.gemini_client = google_genai.Client(
                    api_key=os.environ.get('GEMINI_API_KEY')
                )

            else:
                self.mode = 'heuristic'

        print(f"🤖 Agent Mode: {self.mode.upper()}")

    # =====================================================
    # Main Entry
    # =====================================================

    def propose_design(self, history: List[Tuple[Dict, Dict]]) -> Dict[str, int]:

        if self.mode == 'openai':
            return self._propose_openai(history)

        elif self.mode == 'anthropic':
            return self._propose_anthropic(history)

        elif self.mode == 'gemini':
            return self._propose_gemini(history)

        else:
            return self._propose_heuristic(history)

    # =====================================================
    # OpenAI
    # =====================================================

    def _propose_openai(self, history):

        prompt = self._build_prompt(history)

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
                print(f"💡 GPT proposed: {params}")
                return params

        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")

        return self._propose_heuristic(history)

    # =====================================================
    # Anthropic
    # =====================================================

    def _propose_anthropic(self, history):

        prompt = self._build_prompt(history)

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

    # =====================================================
    # Gemini
    # =====================================================

    def _propose_gemini(self, history):

        prompt = self._build_prompt(history)

        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )

            content = response.text.strip()
            
            # Remove markdown code blocks if present
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

    # =====================================================
    # Heuristic Fallback
    # =====================================================

    def _propose_heuristic(self, history):

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

    # =====================================================
    # Prompt Builder
    # =====================================================

    def _build_prompt(self, history):

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

    # =====================================================
    # History Formatter
    # =====================================================

    def _format_history(self, history):

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

    # =====================================================
    # Validation
    # =====================================================

    def _validate_params(self, params):

        if "PAR" not in params or "BUFFER_DEPTH" not in params:
            return False

        if params["PAR"] not in self.PAR_OPTIONS:
            return False

        if params["BUFFER_DEPTH"] not in self.BUFFER_DEPTH_OPTIONS:
            return False

        return True


# Backward compatibility
def propose_design(history):
    agent = DesignAgent(mode='auto')
    return agent.propose_design(history)