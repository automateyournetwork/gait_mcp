# gait_mcp

A Model Context Protocol (MCP) server for **Git for Artificial Intelligence Tracking (GAIT)**.

GAIT lets AI assistants track conversations, code changes, and context the same way Git tracks source code. This MCP server allows tools like VS Code Copilot, Gemini CLI, and Claude Desktop to persist AI turns, rewind history, branch ideas, and synchronize memory with your working directory.

## What this gives you

- Automatic versioning of AI conversations
- Git-style branching and reverting for AI reasoning
- Persistent memory that survives editor restarts
- First-class support for code artifacts (files + content)
- Cloud-backed remotes via GAITHUB-compatible servers

## Install dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

Upgrade pip and install dependencies:

```bash
pip install -U pip
pip install mcp fastmcp gait-ai
```

## Add the MCP server to VS Code

Add the GAIT MCP server to your VS Code MCP configuration.

**Example (Linux / macOS):**

```json
{
  "servers": {
    "gait": {
      "type": "stdio",
      "command": "/home/johncapobianco/GAITCOPILOT/bin/python",
      "args": [
        "-u",
        "/home/johncapobianco/gait_mcp/gait_mcp.py"
      ],
      "env": {
        "GAITHUB_TOKEN": "your_gaithub_token_here"
      }
    }
  }
}
```

**WSL / Ubuntu on Windows 11**

This configuration worked reliably under WSL:

```json
"gait": {
  "type": "stdio",
  "command": "wsl",
  "args": [
    "/home/johncapobianco/GAITCOPILOT/bin/python",
    "-u",
    "/home/johncapobianco/gait_mcp/gait_mcp.py"
  ],
  "env": {
    "GAITHUB_TOKEN": ""
  }
}
```

## Recommended: Use VS Code Chat Instructions (best results)

Instead of pasting prompts manually, VS Code Copilot supports persistent Chat Instructions via Markdown files.
This is the preferred and most reliable way to use GAIT.

### Step 1: Create the instructions folder

From the root of your repo:

```
.github/instructions/
```

VS Code automatically scans this location by default using the setting:

```
chat.instructionsFilesLocations = { ".github/instructions": true }
```

### Step 2: Create the GAIT instruction file

Create this file:

```
.github/instructions/gait-mcp.md
```

### Step 3: Add GAIT tracking rules

Paste your GAIT MCP rules into that file. For example:

```markdown
GAIT MCP Tracking Rules

These instructions override any default Copilot behavior related to file edits, tool usage, or conversation tracking.

Automatic tracking
Every time you create or modify a file, you MUST call gait_record_turn.

Artifacts
When calling gait_record_turn, you MUST include the full content of every file you created or changed in the artifacts parameter as a list:
[{'path': 'filename', 'content': 'full file contents'}]

Initialization
If GAIT is not initialized in the current directory, you MUST call gait_init before performing any other action.

Revert and resume protocol
If you call gait_revert:

You MUST NOT call gait_record_turn for that action

You MUST immediately call gait_resume

You MUST treat the output of gait_resume as the new ground truth

You MUST ignore any turns that occurred after the reverted commit

Consistency rules
If GAIT tracking is active, you MUST NOT skip recording turns.
If a turn cannot be recorded, you MUST explain why before continuing.
```

### Step 4: Enable the instructions in Copilot

In VS Code:

1. Open Copilot Chat
2. Open the Chat Instructions menu
3. Select `gait-mcp.md`

Once selected, these rules persist for the entire workspace.

You no longer need to paste a prompt for every session.

## Optional: Manual Copilot prompt (fallback)

If you are not using Chat Instructions, you can still bootstrap manually:

```
I want to track our session using the GAIT MCP tool.

First, call gait_init in this directory to ensure tracking is active.

For every task I give you, once you have written or modified code, you must call gait_record_turn.

In that call, include my prompt as user_text, your explanation as assistant_text, and include the full content of any files you created or changed in the artifacts parameter as:
[{'path': 'filename', 'content': 'code'}].

Do you understand these instructions?
```

## Expected workflow

1. Open a project directory
2. Enable GAIT Chat Instructions
3. Ask Copilot to work normally

GAIT automatically:

- Records every meaningful turn
- Tracks file changes as artifacts
- Allows reverting, branching, and resuming AI context

Once initialized, GAIT should track the conversation automatically.

## Notes

- GAIT refuses to initialize at filesystem root by design
- The Copilot chat transcript cannot be erased, but GAIT history can be rewound
- After revert, `gait_resume` is the source of truth
