# CLAUDE.md - GAIT MCP Development Guide

## Project Overview

**GAIT MCP** is a Model Context Protocol (MCP) server that provides Git-like version control for AI conversations. It enables AI assistants (VS Code Copilot, Gemini CLI, Claude Desktop) to track, version, and manage their interactions with users.

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync
uv run python -u gait_mcp.py

# Using pip
python3 -m venv .venv
source .venv/bin/activate
pip install mcp fastmcp gait-ai
python -u gait_mcp.py
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `mcp` | Model Context Protocol base |
| `fastmcp` | FastMCP server framework |
| `gait-ai` | GAIT core library |

---

## Project Structure

```
gait_mcp/
├── gait_mcp.py              # Main MCP server (all tools)
├── pyproject.toml           # Project configuration
├── README.md                # User documentation
├── gait_mcp-MASTERCLASS.md  # Comprehensive guide
├── CLAUDE.md                # This file
├── LICENSE                  # GPL-3.0
├── .gitignore
└── .github/
    └── instructions/
        └── instructions.md  # VS Code Chat Instructions
```

---

## Development Rules

### Rule 1: STDIO Protocol Only

The MCP server communicates via STDIO. **Never write to stdout except protocol messages.**

```python
# CORRECT: Log to stderr
import sys
logging.basicConfig(stream=sys.stderr)

# WRONG: Print to stdout
print("Debug info")  # Breaks MCP protocol!
```

### Rule 2: Structured Error Responses

All tool errors must return structured dicts, never raise unhandled exceptions:

```python
# CORRECT: Return error dict
def _err(msg: str, *, detail: str = "") -> Dict[str, Any]:
    return {"ok": False, "error": msg, "detail": detail}

# WRONG: Raise exception
raise ValueError("Something went wrong")
```

### Rule 3: Never Initialize at Root

GAIT refuses to initialize at filesystem root by design:

```python
def _is_filesystem_root(p: Path) -> bool:
    p = p.resolve()
    return str(p) == str(Path(p.anchor).resolve())
```

This prevents accidental global tracking.

### Rule 4: Sticky Repository Path

The server maintains a sticky path in `~/.gait_mcp_root`:

```python
_STICKY_FILE = Path("~/.gait_mcp_root").expanduser()
```

After `gait_init`, this file stores the repo path for persistence across restarts.

### Rule 5: Wrapper Call Normalization

Some clients (Gemini) use wrapper calling styles. The server normalizes these:

```python
# Accepts these wrapper styles:
# {"args": [...], "kwargs": {...}}
# {"call_args": [...], "call_kwargs": {...}}
# Direct kwargs
```

---

## Tool Categories

### Core Repository Tools

| Tool | Function |
|------|----------|
| `gait_status` | Show repo status |
| `gait_init` | Initialize tracking |
| `gait_branch` | Create branch |
| `gait_checkout` | Switch branch |
| `gait_merge` | Merge branches |
| `gait_log` | Show history |
| `gait_show` | Display commit |

### Turn Recording

| Tool | Function |
|------|----------|
| `gait_record_turn` | Record conversation + artifacts |

**Critical:** Always include full artifacts:
```python
artifacts = [
    {"path": "file.py", "content": "full content"}
]
```

### History Management

| Tool | Function |
|------|----------|
| `gait_revert` | Rewind to previous state |
| `gait_resume` | Sync AI state after revert |
| `gait_summarize_and_squash` | Compress history |

**Rule:** After `gait_revert`, always call `gait_resume`. Never record the revert action.

### Memory Tools

| Tool | Function |
|------|----------|
| `gait_memory` | List pinned items |
| `gait_context` | Build context bundle |
| `gait_pin` | Pin a commit |
| `gait_unpin` | Unpin by index |

### Remote Tools

| Tool | Function |
|------|----------|
| `gait_remote_add` | Add remote URL |
| `gait_remote_list` | List remotes |
| `gait_remote_get` | Get remote URL |
| `gait_repo_create` | Create remote repo |
| `gait_push` | Push to remote |
| `gait_fetch` | Fetch from remote |
| `gait_pull` | Pull and merge |
| `gait_clone` | Clone remote repo |

---

## Code Style

### Python Version

Minimum: Python 3.10 (3.11+ recommended)

```python
# Python 3.10 compatibility shim
if sys.version_info < (3, 11):
    import datetime as dt
    if not hasattr(dt, 'UTC'):
        dt.UTC = dt.timezone.utc
```

### Type Hints

Use type hints for all functions:

```python
def gait_status(path: Optional[str] = None) -> Dict[str, Any]:
    ...
```

### Decorators

All tools use the double-decorator pattern:

```python
@mcp.tool(description="Tool description here.")
@mcp_tool  # Handles errors and wrapper normalization
def gait_tool_name(...) -> Dict[str, Any]:
    ...
```

### Error Handling

```python
@mcp_tool
def gait_example():
    repo, err = _try_repo()
    if err:
        return err  # Returns {"ok": False, "error": "..."}
    assert repo is not None

    # Tool logic here
    return {"ok": True, "result": "..."}
```

---

## Testing Checklist

Before committing changes:

- [ ] Server starts without errors
- [ ] `gait_init` works in a test directory
- [ ] `gait_status` returns correct info
- [ ] `gait_record_turn` creates commits
- [ ] `gait_revert` + `gait_resume` workflow works
- [ ] No stdout pollution (only stderr logging)
- [ ] Error cases return structured responses

---

## MCP Configuration Examples

### VS Code (macOS/Linux)

```json
{
  "servers": {
    "gait": {
      "type": "stdio",
      "command": "/path/to/venv/bin/python",
      "args": ["-u", "/path/to/gait_mcp/gait_mcp.py"],
      "env": {
        "GAITHUB_TOKEN": "your_token_here"
      }
    }
  }
}
```

### VS Code (WSL)

```json
{
  "servers": {
    "gait": {
      "type": "stdio",
      "command": "wsl",
      "args": [
        "/path/to/venv/bin/python",
        "-u",
        "/path/to/gait_mcp/gait_mcp.py"
      ],
      "env": {
        "GAITHUB_TOKEN": ""
      }
    }
  }
}
```

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `GAITHUB_TOKEN` | Remote authentication | For push/pull |
| `GAIT_MCP_STICKY_FILE` | Override sticky path | No |

---

## Git Workflow

### Commit Messages

Follow conventional commits:

```
feat: add gait_squash tool
fix: handle empty HEAD in revert
docs: update MASTERCLASS with diagrams
refactor: extract wrapper normalization
```

### Branch Strategy

- `main` - stable releases
- `develop` - integration branch
- `feature/*` - new features
- `fix/*` - bug fixes

---

## Troubleshooting

### Server Won't Start

```bash
# Check dependencies
uv pip list | grep -E "(mcp|fastmcp|gait)"

# Test import
python -c "from gait.repo import GaitRepo; print('OK')"
```

### "GAIT repo not found"

```bash
# Check sticky file
cat ~/.gait_mcp_root

# Clear and re-init
rm ~/.gait_mcp_root
# Then call gait_init in your project
```

### Protocol Errors

Check for stdout pollution:
```bash
python gait_mcp.py 2>/dev/null | head -1
# Should show nothing (no stdout output on startup)
```

---

## Resources

| Resource | Link |
|----------|------|
| GAIT Core | `pip install gait-ai` |
| FastMCP | `pip install fastmcp` |
| MCP Spec | https://modelcontextprotocol.io |
| MASTERCLASS | `./gait_mcp-MASTERCLASS.md` |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow code style guidelines
4. Add tests for new features
5. Update documentation
6. Submit pull request

---

*Last updated: 2026-01-23*
