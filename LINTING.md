# Linting and Formatting

This project uses Black, isort, and pylint for code formatting, import sorting, and linting.

## Installation

The linters are already installed in the `cuda_projectors` conda environment:
```bash
pip install black isort pylint
```

## Usage

### Format and lint a single file
```bash
# Format
black path/to/file.py
isort path/to/file.py

# Lint
pylint path/to/file.py
```

### Format and lint entire project
```bash
# Format
black .
isort .

# Lint
pylint experiments/ models/ dataset/
```

### Check without modifying (useful for CI)
```bash
black --check .
isort --check-only .
pylint experiments/ models/ dataset/
```

### Format specific directories
```bash
# Format experiments
black experiments/
isort experiments/
pylint experiments/

# Format models
black models/
isort models/
pylint models/

# Format dataset
black dataset/
isort dataset/
pylint dataset/
```

## Configuration

### Black & isort
Configuration is stored in `pyproject.toml`:
- Line length: 100 characters
- Target Python version: 3.9
- isort profile: black (ensures compatibility)

### Pylint (Google Style)
Configuration is stored in `.pylintrc`:
- Google-style naming conventions
- Line length: 100 characters (matches Black)
- Relaxed rules for research code:
  - Allows short variable names (x, y, z, H, W, etc.)
  - No mandatory docstrings (though recommended)
  - Increased limits for function arguments and complexity
  - Allows TODO comments

**Common pylint scores:**
- 10.0: Perfect (rare in practice)
- 8.0-9.9: Excellent
- 7.0-7.9: Good
- 6.0-6.9: Acceptable
- <6.0: Needs improvement

## Recommended Workflow

1. **Before committing:**
   ```bash
   # Format code
   black .
   isort .
   
   # Check for issues
   pylint experiments/ models/ dataset/
   ```

2. **Fix pylint warnings** (aim for score > 7.0)

3. **Commit your changes**

## Pre-commit Hook (Optional)

To automatically format code before committing, you can set up a pre-commit hook:

```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.11.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 6.1.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/pylint
    rev: v3.3.9
    hooks:
      - id: pylint
        args: [--rcfile=.pylintrc]
```

Then run:
```bash
pre-commit install
```

## Ignoring Specific Warnings

If you need to ignore a specific pylint warning in code:

```python
# Disable for a single line
result = some_function()  # pylint: disable=invalid-name

# Disable for a block
# pylint: disable=too-many-locals
def complex_function():
    # ... lots of local variables ...
    pass
# pylint: enable=too-many-locals
```

