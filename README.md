# Agentic Coding Assistant

**Deep Agent with Sub-agent Spawning, TODO List Management, and Pluggable File System**

A general-purpose deep agent framework built on [LangGraph](https://langchain-ai.github.io/langgraph/) that provides hierarchical task delegation through sub-agents, intelligent file system operations, and autonomous workflow management.

## 🌟 Features

### Core Framework (`src/deepagents`)

- **🔄 Sub-agent Spawning**: Delegate complex, multi-step tasks to specialized ephemeral agents
- **📝 TODO List Management**: Built-in task tracking and planning capabilities
- **📁 Pluggable File System**: Multiple backend options (State, Store, Filesystem, Composite)
- **🛠️ Rich Tool Set**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `delete_file`
- **🔌 Middleware Architecture**: Extensible middleware for filesystem, summarization, prompt caching, and more
- **💾 Multiple Storage Backends**: Choose between in-memory state, persistent storage, or hybrid approaches

### Coding Assistant Application (`src/coding`)

A specialized Python code analysis and refactoring agent with:

- **🔍 Impact Analysis**:
  - **SPEED Mode**: AST-based static analysis (~5s for 10K lines)
  - **PRECISION Mode**: Pyright LSP type checking
- **🔧 Autonomous Refactoring**: Self-healing with up to 3 retry attempts
- **✅ Test Generation**: Automatic pytest test creation
- **📚 Documentation Sync**: Keep docstrings and README files updated
- **🌐 Web Search**: Tavily integration for external information
- **⚡ Performance Optimizations**:
  - File and analysis result caching
  - Parallel processing for multi-file operations
  - Context explosion prevention
  - Token limit checking

## 📦 Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepagent.git
cd deepagent

# Install core package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Coding Assistant Installation

```bash
# Install with coding assistant dependencies
pip install -e ".[coding]"
```

## 🚀 Quick Start

### Using the Core Framework

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Create a basic agent with file system capabilities
agent = create_deep_agent(
    model="anthropic/claude-sonnet-4",
    system_prompt="You are a helpful assistant.",
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Create a hello.txt file"}]
})
```

### Using the Coding Assistant

```bash
# Set up environment variables
cd src/coding
cp .env.example .env  # Edit with your API keys

# Run the CLI interface
python run_cli.py
```

Example session:
```
🧑 You: Analyze the impact of changing the calculate_total function in utils.py

🤖 Assistant: I'll analyze the impact using SPEED mode...
[Analysis results showing callers, dependencies, and imports]
```

### Creating Custom Sub-agents

```python
from deepagents import create_deep_agent, SubAgent

# Define a specialized sub-agent
code_reviewer = {
    "name": "code-reviewer",
    "description": "Reviews code for best practices and potential issues",
    "system_prompt": "You are an expert code reviewer...",
    "tools": [],  # Uses parent agent's tools by default
}

# Create agent with sub-agent
agent = create_deep_agent(
    model="anthropic/claude-sonnet-4",
    subagents=[code_reviewer],
)
```

## 🏗️ Architecture

### Core Components

```
deepagents/
├── graph.py              # Agent factory and configuration
├── backends/
│   ├── protocol.py       # Backend interface protocol
│   ├── state.py          # LangGraph state backend
│   ├── store.py          # Persistent store backend
│   ├── filesystem.py     # Real filesystem backend
│   └── composite.py      # Hybrid backend routing
└── middleware/
    ├── filesystem.py     # File operations middleware
    ├── subagents.py      # Sub-agent spawning
    └── patch_tool_calls.py # Tool call correction
```

### Coding Assistant Components

```
coding/
├── coding_agent.py       # Main agent definition
├── optimizations.py      # Performance utilities
├── run_cli.py           # CLI interface
└── test_verification.py  # Testing utilities
```

## 📖 Documentation

### File System Tools

- **`ls(path)`**: List files in directory
- **`read_file(file_path, offset=0, limit=2000)`**: Read file contents with line numbers
- **`write_file(file_path, content)`**: Create new file
- **`edit_file(file_path, old_string, new_string, replace_all=False)`**: Edit existing file
- **`glob(pattern, path=".")`**: Find files matching pattern
- **`grep(pattern, path=".", glob=None, output_mode="files_with_matches")`**: Search in files
- **`delete_file(file_path)`**: Delete file (requires approval)

### Sub-agent Usage

The `task` tool spawns ephemeral sub-agents for complex tasks:

```python
# Agent automatically uses task tool for complex operations
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Research Python async patterns and create a comprehensive guide"
    }]
})
```

### Backend Options

#### 1. State Backend (Default)
Stores files in LangGraph state - temporary, session-scoped.

```python
from deepagents.backends import StateBackend

agent = create_deep_agent(backend=StateBackend)
```

#### 2. Store Backend
Persistent storage using LangGraph Store.

```python
from deepagents.backends import StoreBackend

agent = create_deep_agent(
    backend=StoreBackend,
    store=your_store_instance
)
```

#### 3. Filesystem Backend
Reads/writes to actual filesystem.

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/workspace")
)
```

#### 4. Composite Backend
Hybrid approach with route-based selection.

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/memories/": StoreBackend(),  # Persistent
        "/tmp/": StateBackend(),        # Temporary
    }
)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional
MODEL=moonshotai/kimi-k2-0905
TAVILY_API_KEY=your_tavily_key  # For web search
LOG_LEVEL=INFO
WORKSPACE=/path/to/workspace

# Performance tuning
CACHE_MAX_SIZE=200
CACHE_MAX_MEMORY_MB=50
MAX_FILE_LIST=50
MAX_WORKERS=8
```

### Coding Agent Configuration

```python
from coding_agent import agent

# Configure interrupt behavior
agent = create_deep_agent(
    interrupt_on={
        "delete_file": True,           # Require approval
        "write_file": False,           # Auto-approve
        "change_project_directory": False,
    }
)
```

## 🔬 Advanced Usage

### Impact Analysis Modes

```python
from coding_agent import analyze_impact

# Fast static analysis (AST-based)
result = analyze_impact(
    file_path="src/utils.py",
    function_or_class="calculate_total",
    mode="SPEED"
)

# Precise type checking (Pyright LSP)
result = analyze_impact(
    file_path="src/utils.py",
    function_or_class="calculate_total",
    mode="PRECISION"  # Auto-fallback to SPEED on failure
)
```

### Parallel File Processing

```python
from coding_agent import analyze_multiple_files

# Analyzes files in parallel (3+ files use 8 workers)
result = analyze_multiple_files(
    file_paths=["file1.py", "file2.py", "file3.py"],
    function_or_class="MyClass",
    mode="SPEED"
)
```

### Web Search Integration

```python
from coding_agent import search_web

# Search for external information
result = search_web(
    query="Python async best practices 2024",
    max_results=5
)
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_backends.py
```

## 📊 Performance

- **File Caching**: LRU cache with configurable size and memory limits
- **Analysis Caching**: Results cached by file path, target, and modification time
- **Parallel Processing**: ThreadPoolExecutor for multi-file operations
- **Context Management**: Automatic truncation and summarization
- **Token Optimization**: Prompt caching with Anthropic

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain)
- Type checking with [Pyright](https://github.com/microsoft/pyright)
- Web search powered by [Tavily](https://tavily.com)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/deepagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/deepagent/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/deepagent/wiki)

## 🗺️ Roadmap

- [ ] Additional language support (JavaScript, TypeScript, Java)
- [ ] Visual Studio Code extension
- [ ] Web UI for agent interaction
- [ ] Enhanced monitoring and observability
- [ ] Multi-agent collaboration patterns
- [ ] Custom middleware examples and templates

---

**Made with ❤️ by the DeepAgent Team**
