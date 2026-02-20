"""
LLM-Powered Design Space Exploration Agent

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google Gemini
- Fallback to heuristic search if no API key
"""

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
                # Initialize OpenAI client for v2.x API
                self.openai_client = openai.OpenAI(
                    api_key=os.environ.get('OPENAI_API_KEY')
                )

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
        
        # Handle explicit mode selection
        elif mode == 'openai' and OPENAI_AVAILABLE:
            if os.environ.get('OPENAI_API_KEY'):
                self.openai_client = openai.OpenAI(
                    api_key=os.environ.get('OPENAI_API_KEY')
                )
            else:
                print("[WARN] OPENAI_API_KEY not set, falling back to heuristic")
                self.mode = 'heuristic'
        
        elif mode == 'anthropic' and ANTHROPIC_AVAILABLE:
            if os.environ.get('ANTHROPIC_API_KEY'):
                self.anthropic_client = anthropic.Anthropic(
                    api_key=os.environ.get('ANTHROPIC_API_KEY')
                )
            else:
                print("[WARN] ANTHROPIC_API_KEY not set, falling back to heuristic")
                self.mode = 'heuristic'
        
        elif mode == 'gemini' and GEMINI_AVAILABLE:
            if os.environ.get('GEMINI_API_KEY'):
                self.gemini_client = google_genai.Client(
                    api_key=os.environ.get('GEMINI_API_KEY')
                )
            else:
                print("[WARN] GEMINI_API_KEY not set, falling back to heuristic")
                self.mode = 'heuristic'

        # Safe print for Windows console
        try:
            print(f"🤖 Agent Mode: {self.mode.upper()}")
        except UnicodeEncodeError:
            print(f"[AI] Agent Mode: {self.mode.upper()}")

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
            # Use new OpenAI v2.x API
            response = self.openai_client.chat.completions.create(
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
                try:
                    print(f"💡 GPT proposed: {params}")
                except UnicodeEncodeError:
                    print(f"[IDEA] GPT proposed: {params}")
                return params

        except Exception as e:
            try:
                print(f"⚠️ OpenAI error: {e}")
            except UnicodeEncodeError:
                print(f"[WARN] OpenAI error: {e}")

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
                try:
                    print(f"💡 Claude proposed: {params}")
                except UnicodeEncodeError:
                    print(f"[IDEA] Claude proposed: {params}")
                return params

        except Exception as e:
            try:
                print(f"⚠️ Anthropic error: {e}")
            except UnicodeEncodeError:
                print(f"[WARN] Anthropic error: {e}")

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
                try:
                    print(f"💡 Gemini proposed: {params}")
                except UnicodeEncodeError:
                    print(f"[IDEA] Gemini proposed: {params}")
                return params

        except Exception as e:
            try:
                print(f"⚠️ Gemini error: {e}")
            except UnicodeEncodeError:
                print(f"[WARN] Gemini error: {e}")

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