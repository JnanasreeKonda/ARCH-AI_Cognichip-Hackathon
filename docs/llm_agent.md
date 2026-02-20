# LLM Agent Implementation

## Overview

The LLM Agent is the core intelligence of ARCH-AI, using Large Language Models to propose optimal design parameters based on exploration history.

## Architecture

### DesignAgent Class

The `DesignAgent` class provides a unified interface for multiple LLM providers:

```python
class DesignAgent:
    def __init__(self, mode='auto'):
        # Auto-detect or use specified mode
        # Supports: 'openai', 'anthropic', 'gemini', 'heuristic', 'auto'
    
    def propose_design(self, history):
        # Main entry point
        # Routes to appropriate LLM or heuristic
```

## LLM Providers

### 1. OpenAI GPT-4

**Implementation**: `_propose_openai(history)`

**API**: OpenAI Chat Completions API

**Model**: `gpt-4`

**Configuration**:
- Temperature: 0.7 (balanced creativity)
- Max tokens: 100 (JSON response)
- System prompt: "Respond with valid JSON only"

**Prompt Structure**:
```
You are a hardware design optimization expert.

SEARCH SPACE:
- PAR: [1, 2, 4, 8, 16, 32]
- BUFFER_DEPTH: [256, 512, 1024, 2048]

OBJECTIVE: Minimize AEP = total_cells + 0.5 × (total_cells / throughput)

EXPLORATION HISTORY:
[Formatted table]

INSTRUCTIONS:
1. Analyze patterns
2. Identify promising regions
3. Balance exploration vs exploitation
4. Propose next design

Respond with JSON: {"PAR": <value>, "BUFFER_DEPTH": <value>}
```

### 2. Anthropic Claude

**Implementation**: `_propose_anthropic(history)`

**API**: Anthropic Messages API

**Model**: `claude-3-5-sonnet-20241022`

**Configuration**:
- Temperature: 0.7
- Max tokens: 100

**Similar prompt structure to OpenAI**

### 3. Google Gemini

**Implementation**: `_propose_gemini(history)`

**API**: Google Generative AI API

**Model**: `gemini-pro`

**Configuration**:
- Temperature: 0.7
- Max output tokens: 100

### 4. Heuristic Fallback

**Implementation**: `_propose_heuristic(history)`

**Strategy**: Rule-based search when LLMs unavailable

**Phases**:

1. **Initial Exploration** (first 4 iterations):
   - Iteration 0: PAR=2, DEPTH=1024 (baseline)
   - Iteration 1: PAR=4, DEPTH=1024 (moderate)
   - Iteration 2: PAR=1, DEPTH=512 (minimal area)
   - Iteration 3: PAR=8, DEPTH=2048 (high performance)

2. **Exploitation** (70% probability):
   - Find best design so far
   - Explore neighbors in parameter space
   - Try adjacent PAR values
   - Try adjacent BUFFER_DEPTH values
   - Filter out already-tried combinations

3. **Exploration** (30% probability):
   - Random selection from untried combinations
   - Ensures full design space coverage

4. **Refinement** (all combinations tried):
   - Return best design parameters
   - Local optimization

## History Formatting

### Format for LLM

The `_format_history()` method converts exploration history to a readable table:

```
Iteration | PAR | DEPTH | Cells | FFs | Logic | Objective
----------------------------------------------------------
    1     |  2  |  512  |  295  | 75  |  220  |   368.8
    2     |  4  | 1024  |  400  | 140 |  260  |   450.0
    ...

BEST DESIGN SO FAR:
  PAR=2, DEPTH=512
  Objective=368.8
```

**Features**:
- Last 10 designs only (reduces token usage)
- Clear tabular format
- Best design highlighted
- Objective values formatted

## Parameter Validation

### Validation Rules

The `_validate_params()` method ensures proposed parameters are valid:

```python
def _validate_params(self, params):
    # Check required keys
    if 'PAR' not in params or 'BUFFER_DEPTH' not in params:
        return False
    
    # Check PAR in valid range
    if params['PAR'] not in self.PAR_OPTIONS:
        return False
    
    # Check BUFFER_DEPTH in valid range
    if params['BUFFER_DEPTH'] not in self.BUFFER_DEPTH_OPTIONS:
        return False
    
    return True
```

**Valid Ranges**:
- PAR: {1, 2, 4, 8, 16, 32}
- BUFFER_DEPTH: {256, 512, 1024, 2048}

**Invalid Response Handling**:
- If LLM returns invalid parameters, fallback to heuristic
- Log warning message
- Continue optimization

## Auto-Detection

### Mode Selection

The `mode='auto'` option automatically selects the best available LLM:

```python
if mode == 'auto':
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        self.mode = 'openai'
    elif ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
        self.mode = 'anthropic'
    elif GEMINI_API_KEY and GEMINI_AVAILABLE:
        self.mode = 'gemini'
    else:
        self.mode = 'heuristic'
```

**Priority Order**:
1. OpenAI (if available)
2. Anthropic (if OpenAI unavailable)
3. Gemini (if others unavailable)
4. Heuristic (if no LLMs available)

## Error Handling

### LLM API Errors

All LLM methods include error handling:

```python
try:
    response = llm_api_call(...)
    params = parse_response(response)
    if validate_params(params):
        return params
except Exception as e:
    print(f"LLM error: {e}")
    return self._propose_heuristic(history)
```

**Fallback Strategy**:
- Catch all exceptions
- Log error message
- Fallback to heuristic search
- Continue optimization

### Response Parsing

**JSON Parsing**:
```python
try:
    content = response.strip()
    # Remove markdown code blocks if present
    if content.startswith('```'):
        content = content.split('```')[1]
        if content.startswith('json'):
            content = content[4:]
    params = json.loads(content)
except json.JSONDecodeError:
    # Fallback to heuristic
    return self._propose_heuristic(history)
```

## Performance Considerations

### Token Usage Optimization

- **History Limiting**: Only last 10 designs sent to LLM
- **Concise Prompts**: Minimal but complete information
- **Short Responses**: Max 100 tokens for JSON response

### Latency

- **API Calls**: Each iteration requires one LLM API call
- **Timeout Handling**: 30-second timeout for API calls
- **Caching**: Future enhancement could cache similar requests

### Cost Management

- **Token Efficiency**: Minimal prompt size
- **Model Selection**: Use cost-effective models when possible
- **Fallback**: Heuristic search reduces API costs

## Backward Compatibility

### Function Interface

For backward compatibility, a standalone function is provided:

```python
def propose_design(history):
    """Backward compatibility function"""
    agent = DesignAgent(mode='auto')
    return agent.propose_design(history)
```

This allows existing code to use:
```python
from llm.llm_agent import propose_design
params = propose_design(history)
```

## Future Enhancements

### Potential Improvements

1. **Prompt Optimization**: Fine-tune prompts for better results
2. **Few-Shot Learning**: Include example designs in prompt
3. **Chain-of-Thought**: Ask LLM to explain reasoning
4. **Ensemble Methods**: Combine multiple LLM proposals
5. **Caching**: Cache similar design proposals
6. **Fine-Tuning**: Fine-tune models on hardware design data

### Advanced Features

1. **Multi-Objective**: Explicitly handle multiple objectives
2. **Constraint Awareness**: Include constraint information in prompt
3. **Design Patterns**: Learn from successful design patterns
4. **Transfer Learning**: Use knowledge from similar designs
