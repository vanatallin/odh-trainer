# Model Selection Guide

This guide helps you choose the right small language model for agentic RL training.

## Supported Models

### TinyLlama-1.1B (Default) ⭐

**Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**Specifications**:
- Parameters: 1.1B
- Architecture: Llama 2
- Context Length: 2048 tokens
- GPU Memory: ~4GB (FP16), ~2GB (INT8)
- Training Memory: ~8GB (with gradients & optimizer)

**Pros**:
- ✅ Fast training iterations
- ✅ Fits in 8GB GPU memory
- ✅ Stable for RL training
- ✅ Good quality for its size
- ✅ Pre-trained on chat data
- ✅ Open source (Apache 2.0)

**Cons**:
- ⚠️ Limited capabilities vs larger models
- ⚠️ Shorter context window

**Best For**:
- Experimentation and research
- Learning RL/RLHF techniques
- Limited GPU resources
- Fast prototyping

**Configuration**:
```yaml
env:
  - name: MODEL_NAME
    value: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
resourcesPerNode:
  requests:
    nvidia.com/gpu: 1
    memory: "8Gi"
```

---

### Phi-2 (Microsoft)

**Model**: `microsoft/phi-2`

**Specifications**:
- Parameters: 2.7B
- Architecture: Transformer
- Context Length: 2048 tokens
- GPU Memory: ~8GB (FP16)
- Training Memory: ~16GB

**Pros**:
- ✅ Better quality than TinyLlama
- ✅ Strong reasoning abilities
- ✅ Good instruction following
- ✅ Open source (MIT)

**Cons**:
- ⚠️ Requires more GPU memory
- ⚠️ Slower training than TinyLlama

**Best For**:
- Production RL applications
- Better quality requirements
- 16GB+ GPU available

**Configuration**:
```yaml
env:
  - name: MODEL_NAME
    value: "microsoft/phi-2"
resourcesPerNode:
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
```

---

### Mistral-7B

**Model**: `mistralai/Mistral-7B-v0.1`

**Specifications**:
- Parameters: 7B
- Architecture: Transformer with sliding window attention
- Context Length: 8192 tokens (4x longer!)
- GPU Memory: ~16GB (FP16)
- Training Memory: ~24-32GB

**Pros**:
- ✅ High quality outputs
- ✅ Long context window
- ✅ Strong performance
- ✅ Open source (Apache 2.0)

**Cons**:
- ⚠️ Requires A100 or similar GPU
- ⚠️ Slower training
- ⚠️ Higher costs

**Best For**:
- Production deployments
- Complex reasoning tasks
- Long-form generation
- When quality is critical

**Configuration**:
```yaml
env:
  - name: MODEL_NAME
    value: "mistralai/Mistral-7B-v0.1"
resourcesPerNode:
  requests:
    nvidia.com/gpu: 1
    memory: "32Gi"
```

---

## Comparison Table

| Model | Params | GPU RAM | Training Speed | Quality | Context | License |
|-------|--------|---------|----------------|---------|---------|---------|
| **TinyLlama** | 1.1B | 8GB | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | 2K | Apache 2.0 |
| **Phi-2** | 2.7B | 16GB | ⚡⚡ Medium | ⭐⭐⭐⭐ Better | 2K | MIT |
| **Mistral-7B** | 7B | 32GB | ⚡ Slow | ⭐⭐⭐⭐⭐ Best | 8K | Apache 2.0 |

## RL Training Considerations

### Model Size and RL Stability

Smaller models often train more stably with RL because:
- Faster convergence
- Less catastrophic forgetting
- Easier to debug
- More exploration possible

**Recommendation**: Start with TinyLlama for RL experiments, then scale up.

### Memory Requirements

RL training requires extra memory for:
- Policy model (main)
- Reference model (frozen copy for KL divergence)
- Gradients and optimizer states
- Reward computation buffers

**Formula**: Training Memory ≈ 2.5 × Model Size (FP16)

**Examples**:
- TinyLlama (1.1B): 2.5 × 4GB = **10GB** → Use 8GB GPU
- Phi-2 (2.7B): 2.5 × 8GB = **20GB** → Use 16GB GPU
- Mistral-7B (7B): 2.5 × 16GB = **40GB** → Use 32GB GPU

### GPU Recommendations

| Budget | GPU | Best Model |
|--------|-----|------------|
| Low | T4 (16GB) | TinyLlama |
| Medium | L4 (24GB) / RTX 4090 | Phi-2 |
| High | A100 (40GB/80GB) | Mistral-7B |

## Using Custom Models

You can use any HuggingFace causal language model:

```yaml
env:
  - name: MODEL_NAME
    value: "your-org/your-model"
```

**Requirements**:
1. Model must be compatible with `AutoModelForCausalLM`
2. Must have a tokenizer
3. Should support FP16 training
4. Total memory < available GPU RAM

**Examples of Other Compatible Models**:
- `facebook/opt-1.3b` (1.3B params)
- `EleutherAI/pythia-1.4b` (1.4B params)
- `stabilityai/stablelm-2-1_6b` (1.6B params)
- `google/gemma-2b` (2B params)

## Switching Models

### Via YAML

```yaml
spec:
  trainer:
    env:
      - name: MODEL_NAME
        value: "microsoft/phi-2"  # Change here
    resourcesPerNode:
      requests:
        memory: "16Gi"  # Adjust memory
```

### Via Python SDK

```python
job = AgenticRLTrainingJob(...)
train_job = job.create_train_job(
    model_name="microsoft/phi-2",
    memory_per_node="16Gi"
)
```

## Performance Benchmarks

Based on A100 40GB GPU:

| Model | Training Speed | Episodes/Hour | Memory Usage |
|-------|---------------|---------------|--------------|
| TinyLlama | 🚀 Fast | ~200 | 8GB |
| Phi-2 | ⚡ Medium | ~120 | 16GB |
| Mistral-7B | 🐌 Slow | ~40 | 32GB |

*Note: Actual performance varies with batch size, sequence length, and hardware*

## Model Quality Examples

### Prompt: "Explain photosynthesis to a 5-year-old"

**TinyLlama** (1.1B):
> "Plants eat sunlight! They use sun energy to make food from air and water. The leaves are like little factories."

**Phi-2** (2.7B):
> "Photosynthesis is how plants make their own food. Plants have special parts in their leaves called chlorophyll that capture sunlight. They use this energy to turn water from roots and carbon dioxide from air into sugar, which is their food."

**Mistral-7B** (7B):
> "Imagine plants are like tiny chefs! They take three ingredients: sunlight (energy), water (from soil), and air (carbon dioxide). Their green leaves are like magic kitchens where they mix these together to make sugar - that's their food! As a bonus, they release oxygen which we breathe."

## Recommendations by Use Case

### Research & Experimentation
- **Use**: TinyLlama
- **Why**: Fast iterations, low cost
- **GPU**: 8GB minimum

### Production Prototypes
- **Use**: Phi-2
- **Why**: Good quality-cost balance
- **GPU**: 16GB recommended

### Production Deployment
- **Use**: Mistral-7B or fine-tuned Phi-2
- **Why**: Best quality
- **GPU**: 32GB+ recommended

### Learning RLHF/PPO
- **Use**: TinyLlama
- **Why**: Fastest feedback loop
- **GPU**: 8GB is enough

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1**: Use smaller model
```yaml
MODEL_NAME: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Solution 2**: Reduce batch size
```yaml
BATCH_SIZE: "2"  # From 4
```

**Solution 3**: Use gradient checkpointing
```python
# In agent.py
self.model.gradient_checkpointing_enable()
```

**Solution 4**: Use INT8 quantization
```python
# In agent.py
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

### Slow Training

**Solution**: Use smaller model or reduce sequence length
```yaml
MAX_RESPONSE_LENGTH: "64"  # From 128
```

### Poor Quality

**Solution**: Use larger model
```yaml
MODEL_NAME: "microsoft/phi-2"
```

## Future Models

Watch for these upcoming small models:
- Llama 3 (small variants)
- Phi-3 (Microsoft)
- Gemma 2 (Google)

## Need Help?

- Check model compatibility: [HuggingFace Models](https://huggingface.co/models?pipeline_tag=text-generation)
- GPU memory calculator: [HuggingFace Model Memory](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
- Open an issue: [GitHub Issues](https://github.com/your-org/odh-trainer/issues)
