# CrescendoAttacker
<img width="512" height="512" alt="dce0d188-c231-4a3a-9b25-81e908950634" class="centered-image" src="https://github.com/user-attachments/assets/be30e128-1d6f-4ad8-beaf-459dc3282435" />

# Crescendo Safety Evaluation Framework

A systematic implementation for evaluating AI model robustness against multi-turn jailbreak attempts, based on the [crescendo attack methodology](https://arxiv.org/abs/2404.01833) research by Mark Russinovich and team.

## Overview

This framework provides automated safety evaluation capabilities to assess how AI models respond to progressively escalating prompt sequences. 

**Key Features:**
- Automated generation of progressive prompt sequences
- Systematic evaluation across multiple conversation turns
- Configurable escalation patterns for comprehensive testing
- API integration for scalable model assessment

## Purpose & Applications

This evaluation framework is designed to support:
- **AI Safety Research**: Systematic assessment of model safeguards
- **Pre-deployment Testing**: Automated evaluation of model robustness 
- **Red Team Assessments**: Comprehensive safety evaluation as part of responsible AI development
- **Research & Development**: Understanding model behavior patterns under adversarial conditions

## Technical Implementation

The framework consists of two main components:

### 1. Prompt Generation (`PromptGenerator.py`)
Generates systematic prompt sequences with configurable escalation patterns:
- Creates structured conversation flows
- Implements progressive boundary testing
- Outputs standardized evaluation datasets (`crescendo_prompts.json`)

### 2. Model Evaluation (`CrescendoAttacker.py`)  
Executes systematic safety evaluations:
- Automated API integration (currently supports OpenAI models)
- Sequential prompt execution with response analysis
- Systematic logging of model responses and behaviors

## Quick Start

1. **Generate Evaluation Dataset**
   ```bash
   python PromptGenerator.py
   ```
   Wait for `crescendo_prompts.json` file generation

2. **Configure Model Testing**
   - Set your API key: `export OPENAI_API_KEY="your-key-here"`
   - Configure target model in `CrescendoAttacker.py`

3. **Execute Safety Evaluation**
   ```bash
   python CrescendoAttacker.py
   ```

## Research Foundation

This implementation is based on the crescendo attack methodology described in ["Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"](https://crescendo-the-multiturn-jailbreak.github.io/) which demonstrates how multi-turn conversations can be used to systematically evaluate AI model safety boundaries.

## Future Development

Planned enhancements to improve evaluation capabilities:
- **Enhanced Error Handling**: Robust failure management during evaluation runs  
- **Persistent Storage**: Comprehensive logging and result analysis capabilities
- **Multi-Model Support**: Easy integration with various AI model APIs
- **Dynamic Escalation**: Intelligent adjustment of escalation patterns based on model responses
- **Automated Analysis**: ML-based interpretation of evaluation results and safety boundary identification

## Ethical Use

This framework is intended for legitimate AI safety research and responsible model evaluation. Users should:
- Only test models they own or have explicit permission to evaluate
- Use findings to improve AI safety and alignment
- Follow responsible disclosure practices for any safety issues discovered
- Comply with all applicable terms of service and ethical guidelines

## Contributing

Contributions are welcome, particularly in areas of:
- Multi-model API integration
- Enhanced evaluation metrics
- Result analysis and visualization
- Documentation and examples

---

*This tool is designed to support the responsible development of safer AI systems through systematic safety evaluation.*
