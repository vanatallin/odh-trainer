# Contributing to Agentic RL Example

Thank you for your interest in contributing! This document provides guidelines for contributing to this example.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/odh-trainer.git
   cd odh-trainer/agentic-rl
   ```

3. **Set up your development environment**
   ```bash
   # Install dependencies locally for testing
   cd student
   pip install -r requirements.txt
   cd ../reward-model
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Making Changes

- Create a new branch for your feature:
  ```bash
  git checkout -b feature/your-feature-name
  ```

- Make your changes following the code style guidelines below

- Test your changes locally:
  ```bash
  # Test student code
  cd student
  python -m pytest tests/  # Add tests as needed

  # Test reward model
  cd ../reward-model
  python server.py  # Verify it starts
  ```

### 2. Code Style

#### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Maximum line length: 100 characters

Example:
```python
def generate_response(
    self,
    prompt: str,
    max_length: int = 128,
) -> Tuple[str, torch.Tensor]:
    """
    Generate a response to a prompt.

    Args:
        prompt: Input prompt
        max_length: Maximum length of response

    Returns:
        Tuple of (response_text, log_probs)
    """
    # Implementation
```

#### YAML

- Use 2 spaces for indentation
- Add comments for non-obvious configurations
- Follow Kubernetes best practices

### 3. Testing

Before submitting, test your changes:

1. **Local testing**:
   ```bash
   # Test imports
   python -c "from agent import PPOAgent; from environment import TextGenerationEnvironment"
   ```

2. **Docker build**:
   ```bash
   make build
   ```

3. **Kubernetes deployment** (if possible):
   ```bash
   make deploy-runtime
   make deploy-job
   make status
   ```

### 4. Documentation

Update documentation for any changes:

- Update README.md if adding new features
- Add entries to troubleshooting.md for common issues
- Update scaling.md for performance improvements
- Add examples to alternative-patterns.md for new patterns

### 5. Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer description if needed

- Additional details
- In bullet points
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(agent): add gradient accumulation support

- Implement gradient accumulation in PPO agent
- Add ACCUMULATION_STEPS environment variable
- Update documentation

fix(reward-model): handle empty responses gracefully

- Add validation for empty response strings
- Return default score of 0.5 for invalid inputs
```

## Areas for Contribution

### High Priority

1. **Performance Optimizations**
   - Batch reward queries
   - Async reward model calls
   - GPU memory optimization

2. **Testing**
   - Unit tests for agent.py
   - Integration tests for training loop
   - Mock reward model for testing

3. **Documentation**
   - More examples
   - Video tutorials
   - Architecture diagrams

### Good First Issues

1. **Add logging improvements**
   - Structured logging
   - Metrics to Prometheus
   - TensorBoard integration

2. **Configuration validation**
   - Validate environment variables
   - Better error messages
   - Config file support

3. **Additional prompts**
   - More diverse prompts for training
   - Domain-specific prompt sets
   - Prompt difficulty levels

### Advanced Features

1. **Reward model improvements**
   - Support for multiple reward models
   - Ensemble reward models
   - Custom reward model training

2. **RL algorithms**
   - Implement A2C/A3C
   - Add TRPO support
   - Implement SAC for continuous actions

3. **Distributed training**
   - Better multi-node support
   - Async PPO implementation
   - Gradient compression

## Pull Request Process

1. **Before submitting**:
   - Ensure all tests pass
   - Update documentation
   - Add/update comments
   - Run linting (if configured)

2. **PR description should include**:
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Any breaking changes
   - Related issues

3. **PR template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why these changes are needed

   ## Changes
   - Change 1
   - Change 2

   ## Testing
   How to test these changes

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] All tests passing
   - [ ] No breaking changes (or documented)
   ```

4. **Review process**:
   - Maintainers will review your PR
   - Address any feedback
   - Once approved, PR will be merged

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## Getting Help

- **Questions**: Open a GitHub issue with the `question` label
- **Bugs**: Open a GitHub issue with the `bug` label
- **Feature requests**: Open a GitHub issue with the `enhancement` label

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions

Thank you for contributing! 🎉
