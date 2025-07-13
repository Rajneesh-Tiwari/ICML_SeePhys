# 🧮 Multi-Pipeline Physics Problem Solver

A sophisticated AI-powered physics problem-solving system designed for **ICML 2025 AI for Math Workshop & Challenge 2 - SeePhys**. This system combines OpenAI and Google Gemini models with iterative refinement and intelligent ensemble methods to achieve state-of-the-art performance on complex physics problems.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   OpenAI        │    │   Gemini        │
│   Pipeline      │    │   Pipeline      │
│   (nb0019)      │    │   (nb0021)      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      │
┌─────────────────┐              │
│   Rejection     │              │
│   Improvement   │              │
│   (nb0020)      │              │
└─────────┬───────┘              │
          │                      │
          ▼                      ▼
┌─────────────────────────────────────┐
│         Ensemble System             │
│      (nb0022 & nb0023)              │
│   • Evaluation & Comparison         │
│   • Intelligent Selection           │
│   • Synthesis when both incorrect   │
└─────────────────────────────────────┘
```

## 🤖 Agentic Architecture (LangGraph)

The **nb0019 OpenAI pipeline** leverages **LangGraph** for sophisticated agentic workflow management:

```
┌─────────────────┐
│  Problem Input  │
└─────────┬───────┘
          ▼
┌─────────────────┐    ┌─────────────────┐
│  Generator      │◄──►│  Evaluator      │
│  Agent          │    │  Agent          │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Solution        │    │ Quality &       │
│ Generation      │    │ Confidence      │
│                 │    │ Assessment      │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────────────────────────┐
│         Decision Agent              │
│  • Accept/Reject Logic              │
│  • Iteration Control               │
│  • Feedback Integration            │
│  • State Management                │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────┐
│ Final Solution  │
│ or Re-iterate   │
└─────────────────┘
```

### Key Agentic Features:
- **🧠 Multi-Agent Collaboration**: Generator, Evaluator, and Decision agents work in concert
- **🔄 Autonomous Iteration**: Agents decide when to regenerate vs accept solutions
- **📊 Intelligent State Management**: LangGraph handles complex workflow states and transitions
- **🎯 Goal-Oriented Behavior**: Each agent optimized for specific physics problem-solving tasks
- **🔗 Agent Communication**: Structured information flow between specialized agents

## 📚 Notebooks Overview

### 1. **nb0019.ipynb** - OpenAI Physics Solver Pipeline
- **Purpose**: Primary OpenAI-based physics problem solver with **LangGraph agentic workflow**
- **Model**: OpenAI o3-2025-04-16
- **Features**:
  - **🤖 LangGraph-based agentic workflow management**
  - **🔄 Multi-agent iterative solution refinement with evaluation**
  - Async batch processing for efficiency
  - Few-shot learning with labeled examples
  - Comprehensive physics domain expertise
  - Multi-iteration regeneration on failures
  - **Agent-driven decision making for solution improvement**

### 2. **nb0020.ipynb** - OpenAI Rejection Improvement
- **Purpose**: Targeted improvement of rejected solutions from the OpenAI pipeline
- **Model**: OpenAI o3-2025-04-16 (generator) + o3-2025-04-16 (evaluator)
- **Features**:
  - Feedback-enhanced regeneration
  - Detailed error analysis and correction
  - Complete generator-evaluator improvement loop
  - Advanced prompt engineering for rejection cases
  - Context injection from previous failures

### 3. **nb0021.ipynb** - Gemini Physics Solver Pipeline  
- **Purpose**: Alternative physics solver using Google's Gemini models
- **Model**: Gemini 2.5 Pro Preview (06-05)
- **Features**:
  - Parallel async architecture to OpenAI pipeline
  - Thinking configuration for enhanced reasoning
  - Same physics domain expertise as OpenAI
  - Optimized for Gemini API patterns
  - Vision processing capabilities

### 4. **nb0022.ipynb** - Multi-Pipeline Ensemble (Cost-Optimized)
- **Purpose**: Intelligent comparison and synthesis of OpenAI vs Gemini results
- **Models**: 
  - **o4-mini-2025-04-16** (evaluation - cost-efficient)
  - **o3-2025-04-16** (synthesis - high-quality)
- **Features**:
  - Three-stage pipeline: Evaluate → Select → Synthesize
  - **Cost optimization using smaller model for evaluation**
  - Automatic synthesis when both pipelines fail
  - Complete audit trail and decision rationale
  - **~50% cost reduction vs performance variant**

### 5. **nb0023.ipynb** - Multi-Pipeline Ensemble (Performance-Optimized)
- **Purpose**: High-performance ensemble using top-tier models throughout
- **Models**: 
  - **o3-2025-04-16** (evaluation - maximum accuracy)
  - **o3-2025-04-16** (synthesis - high-quality)
- **Features**:
  - Premium model quality for all operations
  - Enhanced decision-making capabilities
  - Maximum accuracy for critical applications
  - Same three-stage architecture as nb0022

## 🚀 Quick Start

### Prerequisites
```bash
pip install openai google-genai langchain langgraph pydantic tqdm nest-asyncio
```

### Environment Setup
```bash
# Required API keys
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"

# Optional: Configure paths
export PHYSICS_DATA_PATH="/path/to/seephys/problems"
export IMAGES_BASE_PATH="/path/to/problem/images"
```

### Basic Usage

#### 1. Generate OpenAI Baseline Predictions
```python
# Run nb0019.ipynb
from nb0019 import run_async_solver

results_openai = run_async_solver(
    problems=load_seephys_problems(),
    output_file="openai_results.json",
    max_iterations=3,
    batch_size=25
)
```

#### 2. Improve OpenAI Rejections (Optional)
```python
# Run nb0020.ipynb  
from nb0020 import run_rejection_improvement_pipeline

improved_results = run_rejection_improvement_pipeline(
    results_file_path="openai_results.json",
    output_file_path="openai_improved.json",
    max_iterations=5,
    batch_size=25
)
```

#### 3. Generate Gemini Baseline Predictions
```python
# Run nb0021.ipynb
from nb0021 import run_gemini_async_solver

results_gemini = run_gemini_async_solver(
    problems=load_seephys_problems(),
    output_file="gemini_results.json", 
    max_iterations=3,
    batch_size=25
)
```

#### 4. Ensemble Results
```python
# Cost-Optimized Ensemble (nb0022.ipynb)
from nb0022 import run_comparison

final_results_cost = run_comparison(
    openai_results=improved_results,
    gemini_results=results_gemini,
    output_file="ensemble_cost_optimized.json",
    evaluator_model="o4-mini-2025-04-16",  # Cost-efficient
    generator_model="o3-2025-04-16",       # High-quality synthesis
    batch_size=5
)

# Performance-Optimized Ensemble (nb0023.ipynb)  
from nb0023 import run_comparison

final_results_perf = run_comparison(
    openai_results=improved_results,
    gemini_results=results_gemini,
    output_file="ensemble_performance.json", 
    evaluator_model="o3-2025-04-16",       # Maximum accuracy
    generator_model="o3-2025-04-16",       # High-quality synthesis
    batch_size=5
)
```

## 🎯 Key Features

### Advanced Problem Solving
- **🤖 Agentic Workflow System**: LangGraph-powered multi-agent architecture for intelligent decision making
- **Multi-Model Architecture**: Leverages strengths of both OpenAI and Gemini models
- **🔄 Agent-Driven Iterative Refinement**: Self-improving solutions through autonomous feedback loops
- **Domain Expertise**: Specialized physics knowledge across 8+ subject areas
- **Vision Processing**: Handles problems with diagrams, graphs, and equations
- **Intelligent State Management**: LangGraph state machines for complex workflow orchestration

### Cost Optimization Strategy
- **Smart Model Selection**: o4-mini for evaluation tasks, o3 for generation
- **Async Batch Processing**: Minimize API call overhead
- **Intelligent Synthesis**: Only generate new solutions when both pipelines fail
- **Resume Capability**: Avoid reprocessing completed problems
- **Configurable Batch Sizes**: Balance speed vs rate limits

### Quality Assurance  
- **Dual Evaluation**: Both confidence scoring and expert-level assessment
- **Error Analysis**: Detailed categorization of physics errors
- **Ensemble Intelligence**: Automatic selection of best predictions
- **Synthesis Fallback**: Generate new solutions when both pipelines fail

### Production Ready
- **Comprehensive Logging**: Detailed execution tracking and debugging
- **Error Handling**: Robust failure recovery and timeout protection
- **Structured Output**: Pydantic models for type safety and validation
- **Audit Trails**: Complete decision history for transparency

## 📊 Model Configurations & Cost Analysis

| Notebook | Primary Model | Evaluator Model | Cost Tier | Use Case |
|----------|---------------|-----------------|-----------|----------|
| nb0019 | o3-2025-04-16 | o3-2025-04-16 | High | OpenAI Generation |
| nb0020 | o3-2025-04-16 | o3-2025-04-16 | High | Rejection Improvement |
| nb0021 | gemini-2.5-pro-preview-06-05 | gemini-2.5-pro-preview-06-05 | Medium | Gemini Generation |
| nb0022 | o3-2025-04-16 | **o4-mini-2025-04-16** | **Low** | **Cost-Optimized Ensemble** |
| nb0023 | o3-2025-04-16 | o3-2025-04-16 | High | **Performance Ensemble** |

**Cost Optimization Impact:**
- nb0022 uses o4-mini for evaluation (cheaper, faster)
- nb0023 uses o3 for evaluation (expensive, more accurate)
- **Estimated 40-60% cost reduction** with nb0022 vs nb0023

## 🏆 ICML 2025 SeePhys Challenge

This system is specifically designed for the **ICML 2025 AI for Math Workshop & Challenge 2 - SeePhys**, addressing:

### Challenge Requirements
- **Visual Physics Problems**: Complex diagrams, graphs, and mathematical notation
- **Multi-Domain Coverage**: 8 physics subject areas with varying difficulty
- **Solution Quality**: Both reasoning process and final numerical answers
- **Scalability**: Handle large problem sets efficiently

### Competitive Advantages
- **Dual Pipeline Approach**: Combines OpenAI and Gemini strengths
- **Iterative Refinement**: Self-improving solutions through feedback
- **Cost-Performance Balance**: Choose optimal model configuration for your budget
- **Robust Error Handling**: Minimize failures on challenging problems

## 📋 Output Format

Each pipeline produces structured JSON with comprehensive metadata:

```json
{
  "index": 1,
  "question": "Physics problem text...",
  "prediction": "Complete solution with <think> and <answer> tags",
  "confidence_score": 0.85,
  "quality_score": 0.92,
  "source": "openai|gemini|synthesis",
  "iterations_used": 2,
  "total_generations": 3,
  "final_decision": "accept",
  "comparison_metadata": {
    "decision": "choose_openai",
    "decision_rationale": "OpenAI higher confidence and quality",
    "synthesis_required": false,
    "evaluation_summary": "Both solutions correct, OpenAI selected",
    "openai_score": 0.847,
    "gemini_score": 0.823
  }
}
```

## 🔧 Advanced Configuration

### Physics Domain Subjects (SeePhys Categories)
- **O**: Optics (Basic)
- **OPT**: Optics (Extended/Advanced)  
- **EM**: Electromagnetism
- **CM**: Classical Mechanics
- **TSM**: Thermodynamics & Statistical Mechanics
- **QMIT**: Quantum Mechanics & Information Theory
- **ACG**: Astrophysics, Cosmology & Gravitation
- **AMONP**: Atomic, Molecular, Optical & Nuclear Physics

### Error Categories
- Physics theorem errors
- Condition analysis errors  
- Process understanding errors
- Calculation errors
- Variable relationship errors
- Diagram analysis errors
- Boundary conditions errors

## 📈 Performance Metrics

The system tracks comprehensive metrics for SeePhys evaluation:
- **Acceptance Rate**: Percentage of solutions that pass evaluation
- **Confidence Scores**: Model certainty in predictions
- **Quality Scores**: Expert-level solution assessment  
- **Iteration Efficiency**: Average attempts needed per problem
- **Ensemble Effectiveness**: Rate of synthesis requirement
- **Subject Performance**: Success rates by physics domain
- **Cost Efficiency**: API calls per problem ratio

## 💡 Usage Recommendations

### For Competition Submission
1. **Development Phase**: Use nb0022 (cost-optimized) for rapid iteration
2. **Final Submission**: Use nb0023 (performance-optimized) for maximum accuracy
3. **Large Datasets**: Start with single pipelines (nb0019 or nb0021) before ensemble
4. **Budget Conscious**: Focus on OpenAI pipeline with rejection improvement

### Model Selection Guide
- **nb0022**: Good for development, testing, and budget-constrained scenarios
- **nb0023**: Best for final submissions and maximum accuracy requirements
- **Rejection Improvement**: Essential for maximizing OpenAI pipeline performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/seephys-improvement`)
3. Commit your changes (`git commit -m 'Add SeePhys enhancement'`)
4. Push to the branch (`git push origin feature/seephys-improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ICML 2025 AI for Math Workshop & Challenge 2 - SeePhys** organizers (https://www.codabench.org/competitions/7925/?ref=mlcontests#/pages-tab)
- OpenAI for the o3 and o4-mini model families
- Google for Gemini 2.5 Pro
- LangChain & LangGraph communities

---

**Note**: This system is optimized for the ICML 2025 SeePhys challenge. Ensure you have appropriate API quotas and understand the cost implications before running full-scale experiments. The cost-optimized ensemble (nb0022) is recommended for development and testing phases.
