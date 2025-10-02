# AI Coding Rules - StocketAI

## Conciseness Rule
- Be concise: Eliminate unnecessary words in code comments and documentation
- Be direct: State technical requirements clearly without filler
- Be brief: Use short, clear function and variable names
- Focus on essentials: Include only relevant information in task definitions and code

## Technology Stack
- Use Python 3.12+ exclusively
- Core: vnstock, qlib, pandas, numpy, scipy, scikit-learn
- ML: PyTorch, TensorFlow, LightGBM, XGBoost
- Viz: matplotlib, seaborn, plotly
- Install from source code for vnstock and qlib

## Code Standards
- PEP 8 compliance with 88-char line limit
- Type hints for all functions and methods
- Google-style docstrings for public APIs
- Group imports: standard library, third-party, local
- No lazy imports - load all dependencies at startup
- Single responsibility: functions <50 lines preferred
- No files starting with numbers

## Architecture Rules
- Modular design: loosely-coupled components
- Provider abstraction for data sources (vnstock, otherlib/providerA, otherlib/providerB)
- Qlib format conversion for high-performance processing
- Multi-provider fallback mechanisms
- Extensible interfaces for new data sources

## Notebook Organization Rules
- Jupyter notebooks only for user-executable scripts
- No business logic in notebooks - extract to modules
- Provider-specific organization (vn30/, common/, [provider_name]/)
- Production-ready: full workflow execution, no demos
- Atomic and simple: clear sections, progress tracking, reproducibility

## Environment Rules
- Conda mandatory (no venv/virtualenv)
- Windows 11 with PowerShell
- VSCode with Python extensions and Jupyter support
- UTF-8 with BOM for PowerShell scripts
- Proper Git line ending configuration
- Windows environment with PowerShell does not support && command chaining
- Use separate commands or PowerShell syntax (;) for command chaining

## Testing Rules
- Unit tests for individual functions with edge cases
- Integration tests for component interactions
- No external API testing - focus on business logic
- Meaningful test names and comprehensive edge case coverage

## Documentation Rules
- Concise: eliminate unnecessary words
- Direct: clear technical requirements without filler
- Brief: short, clear function and variable names
- Essential information only in task definitions and code
- Self-documenting code with clear variable names

## Production Requirements
- No demo code: all implementations production-ready
- Full functionality in all deliverables
- Complete workflow execution capability
- No partial implementations
- Comprehensive error handling and logging
- Structured logging with appropriate levels

## Quality Gates
- Code quality: passes flake8, black, mypy
- Functionality: meets all specified requirements
- Testing: comprehensive test suite with passing tests
- Documentation: complete and accurate

## Artifact Management
- Code: Python modules in src/ with proper imports
- Data: organized in data/symbols/{symbol}/
- Models: trained models with metadata in models/