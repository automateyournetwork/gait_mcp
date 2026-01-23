# GAIT MCP - MASTERCLASS

## Git for Artificial Intelligence Tracking - Complete Guide

**Version:** 1.0.0
**Last Updated:** 2026-01-23
**License:** GPL-3.0

---

## Table of Contents

1. [What Are We Doing Here?](#1-what-are-we-doing-here)
2. [How Does It Work?](#2-how-does-it-work)
3. [Why Does It Work?](#3-why-does-it-work)
4. [Why We Choose to Run This Way](#4-why-we-choose-to-run-this-way)
5. [What Are the Other Options?](#5-what-are-the-other-options)
6. [Why This Option Is Better](#6-why-this-option-is-better)
7. [Rollback Plan - When Things Go Wrong](#7-rollback-plan---when-things-go-wrong)
8. [Quick Reference](#quick-reference)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## 1. What Are We Doing Here?

### The Big Picture

**GAIT MCP is version control for AI conversations** - it does for AI interactions what Git does for source code.

### The Problem We Solve

When you work with AI assistants (Claude, Copilot, Gemini), you face a fundamental problem:

```mermaid
graph LR
    subgraph Problem["THE PROBLEM"]
        A["AI Session Starts"] --> B["Great Ideas Generated"]
        B --> C["Code Written"]
        C --> D["Session Ends"]
        D --> E["EVERYTHING LOST"]
    end
    style E fill:#ff6b6b,stroke:#fff
    style A fill:#4a90d9,stroke:#fff
    style B fill:#50c878,stroke:#fff
    style C fill:#50c878,stroke:#fff
    style D fill:#f39c12,stroke:#fff
```

**Without GAIT:**
- AI conversations are ephemeral - they vanish when the session ends
- You cannot rewind to a previous point in the reasoning
- You cannot branch to explore alternative approaches
- You cannot share AI context across different tools or team members
- Code generated has no link to the reasoning that created it

**With GAIT:**
- Every turn (user prompt + AI response) is versioned
- Code artifacts are tracked with their reasoning context
- You can revert, branch, and merge AI thought processes
- Context survives editor restarts
- Cloud sync available via GAITHUB-compatible servers

### The Restaurant Analogy

Think of GAIT like a **restaurant order history system**:

| Without GAIT | With GAIT |
|--------------|-----------|
| Waiter remembers orders in head | Orders written on tickets |
| If waiter forgets, start over | Can review any past order |
| Cannot modify previous orders | Can adjust and resubmit |
| One waiter, one memory | Any waiter can continue |
| No record of what worked | Learn from successful orders |

---

## 2. How Does It Work?

### Architecture Overview

```mermaid
graph TB
    subgraph Clients["AI CLIENTS"]
        VS["VS Code Copilot"]
        GEM["Gemini CLI"]
        CLAUDE["Claude Desktop"]
    end

    subgraph MCP["MCP LAYER"]
        SERVER["gait_mcp.py<br/>FastMCP Server"]
    end

    subgraph GAIT["GAIT CORE"]
        REPO["GaitRepo"]
        TURN["Turn Objects"]
        COMMIT["Commit Graph"]
        MEM["Memory/Pinned Context"]
    end

    subgraph Storage["STORAGE"]
        LOCAL[".gait/ Directory"]
        REMOTE["GAITHUB Remote"]
    end

    VS --> |STDIO| SERVER
    GEM --> |STDIO| SERVER
    CLAUDE --> |STDIO| SERVER

    SERVER --> REPO
    REPO --> TURN
    REPO --> COMMIT
    REPO --> MEM

    COMMIT --> LOCAL
    MEM --> LOCAL
    LOCAL <--> |push/pull| REMOTE

    style VS fill:#4a90d9,stroke:#fff
    style GEM fill:#4a90d9,stroke:#fff
    style CLAUDE fill:#4a90d9,stroke:#fff
    style SERVER fill:#f39c12,stroke:#fff
    style REPO fill:#50c878,stroke:#fff
    style LOCAL fill:#9b59b6,stroke:#fff
    style REMOTE fill:#9b59b6,stroke:#fff
```

### Core Components

#### 1. MCP Server (gait_mcp.py)

The Model Context Protocol server exposes GAIT functionality as tools:

| Tool | Purpose |
|------|---------|
| `gait_init` | Initialize GAIT tracking in a directory |
| `gait_status` | Show repo status (branch, HEAD) |
| `gait_record_turn` | Record a conversation turn with artifacts |
| `gait_log` | Show commit history |
| `gait_show` | Display a specific commit |
| `gait_revert` | Rewind to a previous state |
| `gait_resume` | Sync AI state after revert |
| `gait_branch` | Create a new branch |
| `gait_checkout` | Switch branches |
| `gait_merge` | Merge branches |
| `gait_pin` / `gait_unpin` | Manage pinned memory |
| `gait_push` / `gait_pull` | Sync with remote |

#### 2. Turn Objects

A **Turn** captures one exchange:

```python
Turn = {
    "user": {
        "text": "Create a Python function that..."
    },
    "assistant": {
        "text": "Here's the implementation..."
    },
    "context": {
        "artifacts": [
            {"path": "src/utils.py", "content": "def foo():..."}
        ],
        "pinned_context": {...}
    },
    "model": {"provider": "vscode-copilot"},
    "tokens": 1234
}
```

#### 3. Commit Graph

Like Git, GAIT maintains a directed acyclic graph (DAG) of commits:

```mermaid
gitGraph
    commit id: "init"
    commit id: "turn-1"
    commit id: "turn-2"
    branch feature
    commit id: "turn-3"
    commit id: "turn-4"
    checkout main
    commit id: "turn-5"
    merge feature
```

#### 4. Memory System

The **pinned memory** system lets you mark important turns for context:

```mermaid
graph LR
    subgraph History["COMMIT HISTORY"]
        C1["Commit 1"]
        C2["Commit 2"]
        C3["Commit 3<br/>(pinned)"]
        C4["Commit 4"]
        C5["Commit 5<br/>(pinned)"]
    end

    subgraph Memory["PINNED MEMORY"]
        P1["Pin: C3"]
        P2["Pin: C5"]
    end

    C3 -.-> P1
    C5 -.-> P2

    style C3 fill:#f39c12,stroke:#fff
    style C5 fill:#f39c12,stroke:#fff
    style P1 fill:#9b59b6,stroke:#fff
    style P2 fill:#9b59b6,stroke:#fff
```

### Data Flow: Recording a Turn

```mermaid
sequenceDiagram
    participant User
    participant Copilot as VS Code Copilot
    participant MCP as GAIT MCP Server
    participant Core as GAIT Core
    participant FS as .gait/ Directory

    User->>Copilot: "Create a REST API endpoint"
    Copilot->>Copilot: Generate response + code
    Copilot->>MCP: gait_record_turn(user_text, assistant_text, artifacts)
    MCP->>Core: Turn.v0(...) + repo.record_turn()
    Core->>FS: Write turn object
    Core->>FS: Create commit
    Core->>FS: Update HEAD ref
    FS-->>Core: OK
    Core-->>MCP: commit_id
    MCP-->>Copilot: {"ok": true, "commit": "abc123"}
    Copilot-->>User: [Code + tracking confirmation]
```

### Storage Structure

```
project/
├── .gait/                      # GAIT repository (like .git/)
│   ├── objects/                # Content-addressed storage
│   │   ├── turns/             # Turn objects
│   │   └── commits/           # Commit objects
│   ├── refs/
│   │   ├── heads/             # Branch refs (main, feature, etc.)
│   │   └── memory/            # Memory refs per branch
│   ├── HEAD                   # Current branch pointer
│   └── config                 # Repository configuration
├── src/                       # Your actual code
└── README.md
```

---

## 3. Why Does It Work?

### The Underlying Principles

#### 1. Content-Addressed Storage

Like Git, GAIT uses SHA-based addressing:

```
Turn ID = SHA256(user_text + assistant_text + artifacts + timestamp)
```

This guarantees:
- **Immutability**: Objects never change
- **Deduplication**: Identical content = identical ID
- **Integrity**: Corruption is immediately detectable

#### 2. Append-Only History

GAIT follows an append-only model:

```mermaid
graph LR
    subgraph Append["APPEND-ONLY HISTORY"]
        A["Commit A"] --> B["Commit B"]
        B --> C["Commit C"]
        C --> D["Commit D"]
    end

    subgraph Revert["REVERT = NEW POINTER"]
        D -.-> |"revert to B"| B
    end

    style A fill:#4a90d9,stroke:#fff
    style B fill:#50c878,stroke:#fff
    style C fill:#4a90d9,stroke:#fff
    style D fill:#4a90d9,stroke:#fff
```

**Reverting does not delete history** - it moves the HEAD pointer. Original commits remain accessible.

#### 3. Branching for Exploration

AI problem-solving often requires exploring alternatives:

```mermaid
gitGraph
    commit id: "baseline"
    branch approach-a
    commit id: "try REST"
    commit id: "add auth"
    checkout main
    branch approach-b
    commit id: "try GraphQL"
    commit id: "add subscriptions"
    checkout main
    merge approach-b id: "chose GraphQL"
```

GAIT branches let you:
- Try different approaches without losing work
- Compare AI reasoning paths
- Merge the best solution back

#### 4. Context Preservation

The **artifacts** field solves a critical problem:

| Without Artifacts | With Artifacts |
|-------------------|----------------|
| "I created a file" | Full file content stored |
| Lost on session end | Reproducible forever |
| Cannot diff changes | Full diff capability |
| Reasoning disconnected from code | Code + reasoning linked |

#### 5. Memory Pinning

Not all turns are equally important. Pinning lets you:
- Mark key decisions
- Preserve important context
- Build a curated history for future sessions

---

## 4. Why We Choose to Run This Way

### Design Decisions Explained

#### Decision 1: MCP Protocol (STDIO)

**Why STDIO over HTTP?**

```mermaid
graph TB
    subgraph STDIO["STDIO APPROACH"]
        A1["Simple process spawn"]
        A2["No port conflicts"]
        A3["Direct integration"]
        A4["Secure by default"]
    end

    subgraph HTTP["HTTP APPROACH"]
        B1["Port management needed"]
        B2["Authentication overhead"]
        B3["Network latency"]
        B4["Firewall considerations"]
    end

    style A1 fill:#50c878,stroke:#fff
    style A2 fill:#50c878,stroke:#fff
    style A3 fill:#50c878,stroke:#fff
    style A4 fill:#50c878,stroke:#fff
    style B1 fill:#ff6b6b,stroke:#fff
    style B2 fill:#ff6b6b,stroke:#fff
    style B3 fill:#ff6b6b,stroke:#fff
    style B4 fill:#ff6b6b,stroke:#fff
```

STDIO is simpler, faster, and more secure for local tool integration.

#### Decision 2: FastMCP Framework

**Why FastMCP?**

| Feature | Benefit |
|---------|---------|
| Decorators | Clean tool definition (`@mcp.tool`) |
| Auto-serialization | Handles JSON/dict conversion |
| Error handling | Structured error responses |
| Async support | Non-blocking operations |

#### Decision 3: Local-First with Optional Remote

```mermaid
graph TB
    subgraph Local["LOCAL FIRST"]
        L1["Works offline"]
        L2["No account required"]
        L3["Full control"]
        L4["Privacy by default"]
    end

    subgraph Remote["OPTIONAL REMOTE"]
        R1["Cloud backup"]
        R2["Team sharing"]
        R3["Cross-device sync"]
    end

    Local --> |"opt-in"| Remote

    style L1 fill:#50c878,stroke:#fff
    style L2 fill:#50c878,stroke:#fff
    style L3 fill:#50c878,stroke:#fff
    style L4 fill:#50c878,stroke:#fff
    style R1 fill:#4a90d9,stroke:#fff
    style R2 fill:#4a90d9,stroke:#fff
    style R3 fill:#4a90d9,stroke:#fff
```

#### Decision 4: Git-Like Semantics

**Why mirror Git?**

- Familiar mental model for developers
- Proven concepts (branch, merge, revert)
- Easy to explain and adopt
- Natural mapping of AI workflows

#### Decision 5: Sticky Repository Root

The server remembers the last initialized repo via `~/.gait_mcp_root`:

```python
_STICKY_FILE = Path("~/.gait_mcp_root").expanduser()
```

This ensures:
- Survives editor restarts
- No re-initialization needed
- Works across MCP reconnections

---

## 5. What Are the Other Options?

### Alternative Approaches Comparison

```mermaid
graph TB
    subgraph Options["ALTERNATIVES TO GAIT"]
        OPT1["Manual Copy/Paste"]
        OPT2["Chat Export"]
        OPT3["Database Logging"]
        OPT4["File Append"]
        OPT5["GAIT MCP"]
    end

    style OPT5 fill:#50c878,stroke:#fff
    style OPT1 fill:#ff6b6b,stroke:#fff
    style OPT2 fill:#f39c12,stroke:#fff
    style OPT3 fill:#f39c12,stroke:#fff
    style OPT4 fill:#f39c12,stroke:#fff
```

### Option 1: Manual Copy/Paste

**How it works:** Copy AI responses to notes manually.

| Pros | Cons |
|------|------|
| No setup | Labor-intensive |
| Works anywhere | Easy to forget |
| Human curation | No automation |
| | No version control |
| | Loses code-context link |

### Option 2: Chat Export

**How it works:** Export chat history from the AI tool.

| Pros | Cons |
|------|------|
| Built into some tools | Format varies by tool |
| Complete transcript | No branching/merging |
| | Export-only (no revert) |
| | Artifacts often missing |
| | Tool-specific |

### Option 3: Database Logging

**How it works:** Log all interactions to SQLite/PostgreSQL.

| Pros | Cons |
|------|------|
| Queryable | No version semantics |
| Structured | Separate from code |
| Scalable | Requires DB setup |
| | No branch/merge |
| | Heavy for simple use |

### Option 4: File Append

**How it works:** Append each turn to a text file.

| Pros | Cons |
|------|------|
| Simple | Grows unbounded |
| Readable | No structure |
| No dependencies | No revert capability |
| | No branching |
| | No deduplication |

### Option 5: GAIT MCP (This Solution)

**How it works:** Git-like version control via MCP.

| Pros | Cons |
|------|------|
| Full version control | Requires MCP support |
| Branching and merging | Learning curve |
| Code artifacts tracked | Additional tooling |
| Revert capability | |
| Memory pinning | |
| Remote sync | |
| Tool-agnostic | |

---

## 6. Why This Option Is Better

### Feature Comparison Matrix

| Feature | Copy/Paste | Export | Database | File Append | GAIT |
|---------|:----------:|:------:|:--------:|:-----------:|:----:|
| Automatic tracking | - | - | + | + | + |
| Version history | - | - | + | - | + |
| Branching | - | - | - | - | + |
| Merging | - | - | - | - | + |
| Revert capability | - | - | - | - | + |
| Artifacts tracked | - | +/- | + | - | + |
| Context pinning | - | - | - | - | + |
| Remote sync | - | - | + | - | + |
| Tool-agnostic | + | - | + | + | + |
| Offline-first | + | + | +/- | + | + |
| Familiar semantics | + | + | - | + | + |

### Key Advantages

#### 1. Reproducibility

```mermaid
graph LR
    subgraph GAIT["WITH GAIT"]
        G1["Day 1: Create API"] --> G2["Day 2: Clone repo"]
        G2 --> G3["gait_resume"]
        G3 --> G4["Full context restored"]
    end

    subgraph NoGait["WITHOUT GAIT"]
        N1["Day 1: Create API"] --> N2["Day 2: New session"]
        N2 --> N3["Explain everything again"]
        N3 --> N4["Hope AI remembers"]
    end

    style G4 fill:#50c878,stroke:#fff
    style N4 fill:#ff6b6b,stroke:#fff
```

#### 2. Exploration Without Fear

```mermaid
gitGraph
    commit id: "working-v1"
    branch experimental
    commit id: "risky-change"
    commit id: "broke-everything"
    checkout main
    commit id: "safe-improvement"
```

With GAIT, you can experiment freely knowing you can always revert.

#### 3. Team Collaboration

```mermaid
sequenceDiagram
    participant Dev1 as Developer 1
    participant Remote as GAITHUB
    participant Dev2 as Developer 2

    Dev1->>Remote: gait_push (approach A)
    Dev2->>Remote: gait_fetch
    Dev2->>Dev2: Review AI reasoning
    Dev2->>Dev2: Continue from Dev1's context
    Dev2->>Remote: gait_push (improvements)
    Dev1->>Remote: gait_pull
    Dev1->>Dev1: Merge improvements
```

#### 4. Audit Trail

Every decision is documented:
- What was asked
- What was answered
- What code was generated
- When it happened
- How it relates to other decisions

---

## 7. Rollback Plan - When Things Go Wrong

### Scenario Matrix

| Problem | Solution | Command |
|---------|----------|---------|
| Bad AI suggestion accepted | Revert to previous turn | `gait_revert("HEAD~1")` |
| Lost context after restart | Resume from HEAD | `gait_resume()` |
| Wrong branch active | Checkout correct branch | `gait_checkout("main")` |
| Corrupted local repo | Clone from remote | `gait_clone(...)` |
| MCP server crash | Restart server | Re-run `gait_mcp.py` |
| Sticky root wrong | Delete sticky file | `rm ~/.gait_mcp_root` |

### Recovery Procedures

#### Procedure 1: Revert to Previous State

```mermaid
flowchart TD
    A["Problem: Bad turn recorded"] --> B["Call gait_revert"]
    B --> C["Specify target: HEAD~1 or commit ID"]
    C --> D["GAIT resets HEAD pointer"]
    D --> E["Call gait_resume"]
    E --> F["AI state synchronized"]
    F --> G["Continue from clean state"]

    style A fill:#ff6b6b,stroke:#fff
    style G fill:#50c878,stroke:#fff
```

**Commands:**
```
1. gait_revert(target="HEAD~1", also_memory=True)
2. gait_resume(turns=10)
3. [Continue working]
```

**Critical Rule:** After `gait_revert`, always call `gait_resume`. Never record a turn for the revert action itself.

#### Procedure 2: Recover Lost Context

```mermaid
flowchart TD
    A["Problem: Editor restarted, context lost"] --> B["Check gait_status"]
    B --> C{".gait/ exists?"}
    C -->|Yes| D["Call gait_resume"]
    C -->|No| E["Call gait_init"]
    D --> F["Context restored from history"]
    E --> G["Start fresh"]

    style A fill:#ff6b6b,stroke:#fff
    style F fill:#50c878,stroke:#fff
    style G fill:#f39c12,stroke:#fff
```

#### Procedure 3: Branch Rescue

```mermaid
flowchart TD
    A["Problem: Messed up main branch"] --> B["Create rescue branch"]
    B --> C["gait_branch('rescue', from_commit='abc123')"]
    C --> D["gait_checkout('rescue')"]
    D --> E["Work continues on rescue"]
    E --> F["Later: gait_checkout('main')"]
    F --> G["gait_merge('rescue')"]

    style A fill:#ff6b6b,stroke:#fff
    style G fill:#50c878,stroke:#fff
```

#### Procedure 4: Squash Long History

When history becomes too long:

```mermaid
flowchart TD
    A["Problem: Too many small commits"] --> B["gait_summarize_and_squash"]
    B --> C["last=10, mode='soft'"]
    C --> D["Creates backup ref"]
    D --> E["Squashes into single commit"]
    E --> F["Call gait_resume to sync"]

    style A fill:#f39c12,stroke:#fff
    style F fill:#50c878,stroke:#fff
```

**Safe squash (keeps backup):**
```
gait_summarize_and_squash(last=10, mode="soft")
gait_resume()
```

#### Procedure 5: Remote Recovery

```mermaid
flowchart TD
    A["Problem: Local repo corrupted"] --> B["Delete .gait/ directory"]
    B --> C["gait_clone from GAITHUB"]
    C --> D["gait_checkout desired branch"]
    D --> E["gait_resume"]
    E --> F["Fully recovered"]

    style A fill:#ff6b6b,stroke:#fff
    style F fill:#50c878,stroke:#fff
```

### Emergency Contacts

| Issue Type | Action |
|------------|--------|
| Bug in gait_mcp.py | File issue on GitHub |
| GAIT core library bug | Check gait-ai package issues |
| MCP protocol issue | Check FastMCP documentation |
| GAITHUB remote issue | Verify GAITHUB_TOKEN and server status |

### Prevention Checklist

- [ ] Enable Chat Instructions for automatic tracking
- [ ] Configure GAITHUB remote for backup
- [ ] Regularly push to remote
- [ ] Use soft mode for squashing
- [ ] Never init at filesystem root
- [ ] Always call `gait_resume` after `gait_revert`

---

## Quick Reference

### Essential Commands

| Action | Command |
|--------|---------|
| Initialize | `gait_init(path=".")` |
| Check status | `gait_status()` |
| Record work | `gait_record_turn(user_text, assistant_text, artifacts)` |
| View history | `gait_log(limit=20)` |
| Show commit | `gait_show(commit="HEAD")` |
| Revert | `gait_revert(target="HEAD~1")` |
| Resume | `gait_resume(turns=10)` |
| Create branch | `gait_branch(name="feature")` |
| Switch branch | `gait_checkout(name="main")` |
| Merge | `gait_merge(source="feature")` |

### Configuration Files

| File | Purpose |
|------|---------|
| `~/.gait_mcp_root` | Sticky repository path |
| `.github/instructions/gait-mcp.md` | VS Code Chat Instructions |
| `GAITHUB_TOKEN` env var | Remote authentication |

### MCP Server Configuration

**VS Code (macOS/Linux):**
```json
{
  "servers": {
    "gait": {
      "type": "stdio",
      "command": "/path/to/python",
      "args": ["-u", "/path/to/gait_mcp.py"],
      "env": {
        "GAITHUB_TOKEN": "your_token"
      }
    }
  }
}
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: "GAIT repo not found"

**Cause:** No `.gait/` directory in current or parent paths.

**Solution:**
```
1. Navigate to project directory
2. Call gait_init(path=".")
3. Verify with gait_status()
```

#### Issue: "Refusing to initialize at filesystem root"

**Cause:** Attempted `gait_init` at `/` or equivalent.

**Solution:** Navigate to a project folder first.

#### Issue: Recording creates infinite loops

**Cause:** Recording the "tracked successfully" message.

**Solution:** Never call `gait_record_turn` for:
- Revert operations
- Resume operations
- Status checks
- Tool confirmations

#### Issue: Context not restored after restart

**Cause:** Sticky file missing or wrong path.

**Solution:**
```
1. gait_status(path="/full/path/to/project")
2. This updates the sticky file
3. Future calls auto-discover
```

#### Issue: Artifacts missing from history

**Cause:** `gait_record_turn` called without artifacts.

**Solution:** Always include full file content:
```python
artifacts = [
    {"path": "src/app.py", "content": "full file content..."}
]
gait_record_turn(user_text="...", assistant_text="...", artifacts=artifacts)
```

---

## Conclusion

GAIT MCP transforms AI-assisted development from ephemeral chat sessions into a versioned, reproducible workflow. By applying Git-like semantics to AI conversations, it enables:

- **Persistent memory** across sessions
- **Branching** for exploration
- **Reverting** when things go wrong
- **Collaboration** via remote sync
- **Audit trails** for every decision

The investment in setup pays dividends in productivity, reproducibility, and peace of mind.

---

*GAIT MCP MASTERCLASS - Created by The Professor (Agent #27)*
*For questions or improvements, file an issue on GitHub.*
