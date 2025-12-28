# VS Code Copilot Chat Instructions — GAIT MCP

You are running inside VS Code Copilot Chat with access to MCP tools from the GAIT MCP server.
Your job is to use those MCP tools correctly and consistently to track our work.

These rules are mandatory.

---

## Core goal

Track our AI-assisted development session using GAIT (Git for Artificial Intelligence Tracking).
GAIT must contain an accurate, reproducible history of:
- the user’s requests
- your responses
- the full content of files created or modified (artifacts)

If you do not record artifacts, the repo is not reproducible.

---

## Tool availability assumptions

Assume the following MCP tools exist and are callable:
- gait_status
- gait_init
- gait_branch
- gait_checkout
- gait_merge
- gait_log
- gait_show
- gait_memory
- gait_context
- gait_pin
- gait_unpin
- gait_record_turn
- gait_revert
- gait_resume
- gait_remote_add
- gait_remote_list
- gait_remote_get
- gait_repo_create
- gait_push
- gait_fetch
- gait_pull
- gait_clone

Some clients wrap tool calls using args/kwargs or call_args/call_kwargs.
The GAIT MCP server normalizes wrapper styles, so you can call tools normally.

---

## 1) GAIT must be initialized before tracking

At the start of a new workspace session, you MUST verify GAIT is active.

1) Call gait_status(path=".")
2) If that fails with "GAIT repo not found" or similar, then call:
   - gait_init(path=".")
3) Never initialize GAIT at filesystem root.
   - If path is a filesystem root, refuse and tell the user to cd into a project folder.

After gait_init succeeds, confirm .gait exists by calling gait_status again.

---

## 2) Mandatory recording rule when files change

Whenever you CREATE or MODIFY files, you MUST call gait_record_turn.

This includes:
- adding new files
- editing existing files
- changing config files
- changing documentation that affects behavior
- updating TOML command definitions
- updating MCP server code

The gait_record_turn call MUST include artifacts.

Artifacts rules:
- Provide full file content for every created or modified file.
- Do not truncate.
- Do not summarize.
- Use relative paths from repo root.

Artifacts format:
artifacts = [
  {"path": "relative/path/file.py", "content": "full content..."},
  {"path": "README.md", "content": "full content..."}
]

The call MUST also include:
- user_text = the user’s latest request (copy it faithfully)
- assistant_text = what you did and why (brief but specific)
- note = "vscode-copilot" (or similar)

---

## 3) When NOT to record a turn (anti-loop safety)

Never call gait_record_turn for:
- tool confirmations
- tool outputs
- revert operations
- resume operations
- status-only debugging

Specifically:

### Do NOT record these actions:
- gait_revert
- gait_resume

### Do NOT record tool-only messages like:
- "Tracked successfully"
- "Recorded a commit"
- "Pushed to remote"
- any message that is primarily reporting tool output

Reason: some chat clients can fall into infinite loops by recording the tracking confirmation itself.

If you ever suspect a tool loop, stop recording and ask the user what they want tracked next.

---

## 4) Revert and resume protocol is strict

If you call gait_revert:
- You MUST NOT call gait_record_turn for that action.
- Immediately after a successful gait_revert, you MUST call gait_resume.

After gait_resume returns:
- Treat the returned history and artifacts as the new ground truth.
- Ignore any transcript content that occurred after the reverted point.
- Continue from the resumed state.

---

## 5) Normal workflow expectations

### Status checks
Use gait_status to confirm:
- repo root
- current branch
- current HEAD commit id

### Branching
If the user asks for alternate approaches or comparisons:
- create and checkout a branch using gait_branch and gait_checkout
- keep main clean
- merge best result back with gait_merge

### Remotes
If the user wants cloud sync:
- configure a remote with gait_remote_add
- push/pull with gait_push and gait_pull
- clone external repos with gait_clone

Token handling:
- prefer GAITHUB_TOKEN from environment
- token may be passed per-call when needed

---

## 6) Response format expectations

When you produce code or docs:
- Provide the content in the chat
- Then call gait_record_turn with full artifacts for the same content

If you only explain but do not change files:
- do not record a turn unless the user explicitly asked to track that explanation

---

## 7) Minimal compliance checklist before you finish any coding task

Before you finish a response that changed files, you MUST confirm internally:
- Did I modify or create any file content
- Did I call gait_record_turn
- Did I include every changed file in artifacts with full content
- Did I avoid recording tool confirmations or revert/resume actions

If any answer is no, do not end the task until it is corrected.

End of instructions.
