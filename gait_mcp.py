#!/usr/bin/env python3
"""
MCP Server: GAIT (Git for AI Tracking) for Gemini-CLI, Claude Desktop, and VS Code.

Design goals:
- STDIO MCP: NEVER write to stdout except protocol (FastMCP handles this)
- Log to STDERR only
- Be resilient when GAIT is not initialized yet (return structured errors)
- Enforce: DO NOT init at filesystem root
- Provide GAIT-native revert/reset semantics (optionally also reset memory)
- Provide remote add/list/get + push/fetch/pull/clone + repo create

Gemini-CLI note:
Some Gemini wrappers call tools using one of these styles:
  tool(args=[...], kwargs={...})
  tool(call_args=[...], call_kwargs={...})
So we must accept and unpack those safely.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Logging (stderr only)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

log = logging.getLogger("GaitMCP")

# ---------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    try:
        from fastmcp import FastMCP  # type: ignore
    except Exception as e:
        log.error("FastMCP not found. Install with: pip install mcp")
        raise

# ---------------------------------------------------------------------
# GAIT core
# ---------------------------------------------------------------------
from gait.repo import GaitRepo
from gait.schema import Turn
from gait.tokens import count_turn_tokens
from gait.objects import short_oid
from gait.log import walk_commits

# ---------------------------------------------------------------------
# GAIT remote (your real implementation)
# ---------------------------------------------------------------------
from gait.remote import (
    RemoteSpec,
    remote_add,
    remote_get,
    remote_list,
    push as remote_push,
    fetch as remote_fetch,
    pull as remote_pull,
    clone_into,
    create_repo as remote_create_repo,
)

if sys.version_info < (3, 11):
    # Shim for Python 3.10 compatibility
    import datetime as dt
    if not hasattr(dt, 'UTC'):
        dt.UTC = dt.timezone.utc

# ---------------------------------------------------------------------

mcp = FastMCP("GAIT")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _err(msg: str, *, detail: str = "", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error": msg}
    if detail:
        out["detail"] = detail
    out.update(extra)
    return out


def _is_filesystem_root(p: Path) -> bool:
    p = p.resolve()
    return str(p) == str(Path(p.anchor).resolve())


def _get_gaithub_token(provided: str = "") -> Optional[str]:
    t = (provided or "").strip()
    if t:
        return t
    env = os.environ.get("GAITHUB_TOKEN", "").strip()
    return env or None


def _require_gaithub_token(provided: str = "") -> str:
    tok = _get_gaithub_token(provided)
    if not tok:
        raise RuntimeError("GAITHUB_TOKEN is not set (and token was not provided as an argument).")
    return tok


def _remote_spec(repo: GaitRepo, remote: str, owner: str, repo_name: str) -> RemoteSpec:
    base_url = remote_get(repo, remote)
    return RemoteSpec(base_url=base_url, owner=owner, repo=repo_name, name=remote)


def _resolve_commit_prefix_from_head(repo: GaitRepo, head: str, prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("empty commit prefix")

    cid = head
    seen = set()
    while cid and cid not in seen:
        seen.add(cid)
        if cid.startswith(prefix):
            return cid
        c = repo.get_commit(cid)
        parents = c.get("parents") or []
        cid = parents[0] if parents else ""
    raise ValueError(f"Unknown commit or prefix: {prefix}")


def _resolve_revert_target(repo: GaitRepo, target: str) -> str:
    t = (target or "").strip()
    head = repo.head_commit_id() or ""
    if not head:
        raise ValueError("No HEAD commit to revert")

    if not t:
        t = "HEAD~1"

    m = re.fullmatch(r"HEAD~(\d+)", t.upper())
    if m:
        n = int(m.group(1))
        if n <= 0:
            raise ValueError("HEAD~N must be >= 1")

        cid = head
        for _ in range(n):
            c = repo.get_commit(cid)
            parents = c.get("parents") or []
            cid = parents[0] if parents else ""
            if not cid:
                break
        return cid  # may be "" meaning empty

    return _resolve_commit_prefix_from_head(repo, head, t)


def _unpack_wrapper_call(call_kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Normalize wrapper calling styles into (args, kwargs).

    Supported wrapper payloads:
      - {"args":[...], "kwargs":{...}}
      - {"call_args":[...], "call_kwargs":{...}}

    If neither wrapper exists, treat call_kwargs as real kwargs (minus any stray wrapper keys).
    """
    if "call_args" in call_kwargs or "call_kwargs" in call_kwargs:
        akey, kkey = "call_args", "call_kwargs"
    else:
        akey, kkey = "args", "kwargs"

    inner_args: List[Any] = []
    inner_kwargs: Dict[str, Any] = {}

    if kkey in call_kwargs and isinstance(call_kwargs[kkey], dict):
        inner_kwargs = dict(call_kwargs[kkey])

    if akey in call_kwargs and isinstance(call_kwargs[akey], list):
        inner_args = list(call_kwargs[akey])

    # If wrapper DIDN'T actually use either form, treat outer kwargs as real kwargs.
    if not inner_args and not inner_kwargs:
        outer = dict(call_kwargs)
        outer.pop("args", None)
        outer.pop("kwargs", None)
        outer.pop("call_args", None)
        outer.pop("call_kwargs", None)
        return ([], outer)

    return (inner_args, inner_kwargs)


def _coerce_record_turn_kwargs(
    args2: List[Any],
    kwargs2: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Fix for wrappers that send only one side of a turn, or use alternate names.

    We accept (and normalize) these aliases:
      - user_text: "user" | "input" | "prompt" | "question"
      - assistant_text: "assistant" | "output" | "response" | "answer"
      - note: "message"
      - use_memory_snapshot: "snapshot" | "memory_snapshot"

    We ALSO support positional calling patterns some wrappers may use:
      - gait_record_turn("just user text")
      - gait_record_turn("user text", "assistant text")
    """
    # 1) Positional fallbacks
    # If the tool was invoked positionally, map:
    #   [0] -> user_text, [1] -> assistant_text
    if args2:
        if "user_text" not in kwargs2 and len(args2) >= 1 and isinstance(args2[0], str):
            kwargs2["user_text"] = args2[0]
        if "assistant_text" not in kwargs2 and len(args2) >= 2 and isinstance(args2[1], str):
            kwargs2["assistant_text"] = args2[1]
        # We consumed positional intent; clear args to avoid double-binding
        args2 = []

    # 2) Keyword aliases
    aliases = {
        "user": "user_text",
        "input": "user_text",
        "prompt": "user_text",
        "question": "user_text",
        "assistant": "assistant_text",
        "output": "assistant_text",
        "response": "assistant_text",
        "answer": "assistant_text",
        "message": "note",
        "snapshot": "use_memory_snapshot",
        "memory_snapshot": "use_memory_snapshot",
    }
    for src, dst in aliases.items():
        if dst not in kwargs2 and src in kwargs2:
            kwargs2[dst] = kwargs2.pop(src)

    # 3) Some wrappers send role+text instead of user/assistant fields
    #    e.g. {"role":"user","text":"hi"} or {"role":"assistant","text":"hello"}
    role = (kwargs2.get("role") or "").strip().lower()
    text = kwargs2.get("text")
    if isinstance(text, str) and text.strip():
        if role == "user" and not kwargs2.get("user_text"):
            kwargs2["user_text"] = text
        elif role in ("assistant", "model") and not kwargs2.get("assistant_text"):
            kwargs2["assistant_text"] = text

    # 4) Ensure keys exist (so downstream code is stable)
    kwargs2.setdefault("user_text", "")
    kwargs2.setdefault("assistant_text", "")
    kwargs2.setdefault("note", "gemini-cli")
    kwargs2.setdefault("use_memory_snapshot", True)

    return args2, kwargs2

# ---------------------------------------------------------------------
# Sticky repo root (survives MCP server restarts)
# ---------------------------------------------------------------------
_LAST_REPO_ROOT: Optional[Path] = None
_STICKY_FILE = Path(os.environ.get("GAIT_MCP_STICKY_FILE", "~/.gait_mcp_root")).expanduser()

def _load_sticky_root() -> Optional[Path]:
    try:
        if _STICKY_FILE.is_file():
            txt = _STICKY_FILE.read_text(encoding="utf-8").strip()
            if txt:
                p = Path(txt).expanduser().resolve()
                if p.exists():
                    return p
    except Exception:
        pass
    return None

def _persist_sticky_root(p: Path) -> None:
    try:
        _STICKY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STICKY_FILE.write_text(str(p.resolve()), encoding="utf-8")
    except Exception:
        # never fail tool calls because of this
        pass

def _set_last_root(p: Path) -> None:
    global _LAST_REPO_ROOT
    _LAST_REPO_ROOT = p.resolve()
    _persist_sticky_root(_LAST_REPO_ROOT)

# Load once at import time so restarts remember the last repo
_LAST_REPO_ROOT = _load_sticky_root()

def _discover_repo_from(start: Path) -> Optional[GaitRepo]:
    cur = start.expanduser().resolve()
    for p in (cur, *cur.parents):
        if (p / ".gait").is_dir():
            _set_last_root(p)
            return GaitRepo(root=p)
    return None

def _try_repo(path: Optional[str] = None) -> Tuple[Optional[GaitRepo], Optional[Dict[str, Any]]]:
    try:
        candidates: List[Path] = []

        # 1) explicit path always wins
        if path is not None and str(path).strip():
            candidates.append(Path(path).expanduser().resolve())
        else:
            # 2) in-memory sticky
            if _LAST_REPO_ROOT is not None:
                candidates.append(_LAST_REPO_ROOT)

            # 3) persisted sticky (covers server restarts)
            sticky = _load_sticky_root()
            if sticky is not None:
                candidates.append(sticky)

            # 4) cwd
            candidates.append(Path.cwd())

            # 5) home as last resort
            candidates.append(Path.home())

        for start in candidates:
            repo = _discover_repo_from(start)
            if repo is not None:
                return (repo, None)

        return (
            None,
            _err(
                "GAIT repo not found. Run /gait:init inside a project folder.",
                detail=f"candidates={[str(c.expanduser().resolve()) for c in candidates]} cwd={os.getcwd()} sticky={str(_LAST_REPO_ROOT) if _LAST_REPO_ROOT else ''}",
            ),
        )

    except Exception as e:
        return (None, _err("GAIT repo not found. Run /gait:init inside a project folder.", detail=str(e)))


def mcp_tool(fn):
    """
    Decorator for MCP tools.

    - Accepts wrapper calling styles used by some Gemini wrappers (args/kwargs or call_args/call_kwargs).
    - Preserves normal MCP invocation (positional args permitted).
    - Special-cases gait_record_turn to normalize one-sided turns.
    - Converts exceptions into structured error dicts: {"ok": False, "error": "...", "detail": "..."}.
    """
    @functools.wraps(fn)
    def wrapper(*call_args: Any, **call_kwargs: Any):
        try:
            args2, kwargs2 = _unpack_wrapper_call(call_kwargs)

            # If MCP runtime passed positional args normally, preserve them
            if call_args:
                args2 = list(call_args) + args2

            # Special normalization for gait_record_turn to retain user-only or assistant-only turns
            if fn.__name__ == "gait_record_turn":
                args2, kwargs2 = _coerce_record_turn_kwargs(list(args2), dict(kwargs2))

            return fn(*args2, **kwargs2)

        except Exception as e:
            log.exception("tool failed: %s", fn.__name__)
            return _err(f"{fn.__name__} failed", detail=str(e))

    return wrapper


# ---------------------------------------------------------------------
# Core repo tools
# ---------------------------------------------------------------------
@mcp.tool(description="Show GAIT repo status: root path, current branch, and HEAD commit id.")
@mcp_tool
def gait_status(path: Optional[str] = None) -> Dict[str, Any]:
    repo, err = _try_repo(path)
    if err:
        return err
    assert repo is not None
    return {"ok": True, "root": str(repo.root), "branch": repo.current_branch(), "head": repo.head_commit_id() or ""}


@mcp.tool(description="Initialize GAIT tracking in the given folder (refuses filesystem root). Creates the .gait directory.")
@mcp_tool
def gait_init(path: str = ".") -> Dict[str, Any]:
    root = Path(path).resolve()
    if _is_filesystem_root(root):
        return _err("Refusing to initialize GAIT at filesystem root. cd into a working folder first.", path=str(root))

    repo = GaitRepo(root=root)
    repo.init()
    _set_last_root(root)   # <-- important
    return {"ok": True, "root": str(repo.root), "gait_dir": str(repo.gait_dir)}


@mcp.tool(description="Create a new branch from an optional commit. Optionally inherit pinned memory. Use force=true to reset if exists.")
@mcp_tool
def gait_branch(
    name: str,
    from_commit: Optional[str] = None,
    inherit_memory: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    try:
        repo.create_branch(name, from_commit=from_commit, inherit_memory=inherit_memory)
        return {"ok": True, "created": True, "branch": name}
    except FileExistsError:
        if not force:
            return {"ok": True, "created": False, "branch": name, "note": "already exists (use force=true to reset)"}

        target = from_commit if from_commit is not None else repo.head_commit_id()
        repo.write_ref(name, target or "")
        if inherit_memory:
            repo.write_memory_ref(repo.read_memory_ref(repo.current_branch()), name)

        return {"ok": True, "created": False, "reset": True, "branch": name, "head": target or ""}


@mcp.tool(description="Checkout an existing branch (updates current branch and HEAD).")
@mcp_tool
def gait_checkout(name: str) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    repo.checkout(name)
    return {"ok": True, "branch": repo.current_branch(), "head": repo.head_commit_id() or ""}


@mcp.tool(description="Merge a source branch into the current branch, optionally merging memory.")
@mcp_tool
def gait_merge(source: str, message: str = "", with_memory: bool = False) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    merge_id = repo.merge(source, message=message or "", with_memory=with_memory)
    out: Dict[str, Any] = {"ok": True, "merged": source, "branch": repo.current_branch(), "head": short_oid(merge_id)}
    if with_memory:
        out["memory"] = repo.read_memory_ref(repo.current_branch())
    return out


@mcp.tool(description="List recent commits on the current branch (first-parent walk).")
@mcp_tool
def gait_log(limit: int = 20) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    commits = []
    for c in walk_commits(repo, limit=limit):
        cid = c["_id"]
        parents = c.get("parents") or []
        commits.append(
            {
                "commit": short_oid(cid),
                "id": cid,
                "created_at": c.get("created_at") or "",
                "kind": c.get("kind") or "",
                "message": c.get("message") or "",
                "parents": [short_oid(x) for x in parents],
                "turns": len(c.get("turn_ids") or []),
                "merge": len(parents) > 1,
            }
        )
    return {"ok": True, "branch": repo.current_branch(), "commits": commits}


@mcp.tool(description="Show a commit (HEAD or id/prefix) including recorded turns and code artifacts.")
@mcp_tool
def gait_show(commit: str = "HEAD") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    head = repo.head_commit_id() or ""
    if not head:
        return _err("No commits in this branch yet")

    cid = head if commit in ("", "HEAD", "@") else _resolve_commit_prefix_from_head(repo, head, commit)
    c = repo.get_commit(cid)

    turn_ids = c.get("turn_ids") or []
    turns = []
    for tid in turn_ids:
        t = repo.get_turn(tid)
        # Pull artifacts out of the context metadata
        artifacts = t.get("context", {}).get("artifacts", [])
        
        turns.append(
            {
                "turn_id": tid,
                "user": (t.get("user") or {}).get("text", ""),
                "assistant": (t.get("assistant") or {}).get("text", ""),
                "artifacts": artifacts, # Include the code in the output
            }
        )

    return {
        "ok": True,
        "commit": cid,
        "short": short_oid(cid),
        "created_at": c.get("created_at") or "",
        "kind": c.get("kind") or "",
        "message": c.get("message") or "",
        "turns": turns,
    }

# ---------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------
@mcp.tool(description="List pinned memory items (commit/turn references) for the current branch.")
@mcp_tool
def gait_memory() -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    manifest = repo.get_memory()
    items = []
    for i, it in enumerate(manifest.items, start=1):
        items.append({"index": i, "turn": short_oid(it.turn_id), "commit": short_oid(it.commit_id), "note": it.note})
    return {"ok": True, "branch": repo.current_branch(), "pinned": len(items), "items": items}


@mcp.tool(description="Build a context bundle from pinned memory (full=false is compact; full=true expands).")
@mcp_tool
def gait_context(full: bool = False) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None
    return {"ok": True, "bundle": repo.build_context_bundle(full=full)}


@mcp.tool(description="Pin a commit (or last commit) into GAIT memory with an optional note.")
@mcp_tool
def gait_pin(commit: Optional[str] = None, last: bool = True, note: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.pin_commit(commit, last=last, note=note or "")
    return {"ok": True, "memory_id": mem_id}


@mcp.tool(description="Unpin a memory item by 1-based index from gait_memory.")
@mcp_tool
def gait_unpin(index: int) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    mem_id = repo.unpin_index(index)
    return {"ok": True, "unpinned": index, "memory_id": mem_id}


# ---------------------------------------------------------------------
# Turn recording (auto tracking)
# ---------------------------------------------------------------------
@mcp.tool(description="Record a turn and include code artifacts/files created.")
@mcp_tool
def gait_record_turn(
    user_text: str = "",
    assistant_text: str = "",
    artifacts: Optional[List[Dict[str, str]]] = None, # New: List of {"path": "...", "content": "..."}
    note: str = "vscode-copilot",
    use_memory_snapshot: bool = True,
) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err: return err
    
    # Bundle the artifacts into the turn metadata
    context: Dict[str, Any] = {"artifacts": artifacts or []}
    
    if use_memory_snapshot:
        try:
            context["pinned_context"] = repo.build_context_bundle(full=False)
        except Exception as e:
            context["pinned_context_error"] = str(e)

    # Calculate tokens (ensure your count_turn_tokens handles the extra context)
    tokens = count_turn_tokens(user_text=user_text, assistant_text=assistant_text)

    turn = Turn.v0(
        user_text=user_text,
        assistant_text=assistant_text,
        context=context, # The code is now stored here!
        tools={},
        model={"provider": "vscode-copilot"},
        tokens=tokens,
        visibility="private",
    )
    _, commit_id = repo.record_turn(turn, message=note)
    return {"ok": True, "commit": short_oid(commit_id), "artifacts_tracked": len(artifacts or [])}

# ---------------------------------------------------------------------
# Revert / reset
# ---------------------------------------------------------------------
@mcp.tool(description="Rewind branch history and optionally reset pinned memory. After calling this, you MUST call gait_resume to sync state.")
@mcp_tool
def gait_revert(target: str = "HEAD~1", also_memory: bool = True) -> Dict[str, Any]:
    """Rewind history and sync AI memory. Instructions included for Copilot to skip recording this turn."""
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    branch = repo.current_branch()
    head_before = repo.head_commit_id() or ""
    if not head_before:
        return _err("Nothing to revert (branch has no commits).", branch=branch)

    new_head = _resolve_revert_target(repo, target)

    # Perform the History Reset
    if new_head == "":
        repo.write_ref(branch, "")
        head_after = ""
    else:
        head_after = repo.reset_branch(new_head)

    out: Dict[str, Any] = {
        "ok": True,
        "branch": branch,
        "head_before": short_oid(head_before),
        "head_after": short_oid(head_after) if head_after else "(empty)",
        # CRITICAL: This instruction tells Copilot what to do next
        "instruction": "HISTORY REWOUND. Do NOT call gait_record_turn for this action. Immediately call gait_resume() to see the new state of the code and memory."
    }

    # Keep the Memory Logic! 
    # This ensures the AI's 'pinned' context matches the new point in time.
    if also_memory:
        try:
            old_mem = repo.read_memory_ref(branch)
            # Reset memory to match the state of the repo at the new head
            new_mem = repo.reset_memory_to_commit(branch, repo.head_commit_id())
            out["memory_status"] = "Memory synced to new HEAD"
            out["memory_id"] = new_mem
        except Exception as e:
            out["memory_error"] = f"Could not sync memory: {str(e)}"

    return out

# ---------------------------------------------------------------------
# Remote tools
# ---------------------------------------------------------------------
@mcp.tool(description="Add a named remote (e.g., origin) pointing to a GAITHUB-compatible base URL.")
@mcp_tool
def gait_remote_add(name: str, url: str) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    remote_add(repo, name, url)
    return {"ok": True, "remote": name, "url": url}


@mcp.tool(description="List configured remotes (optionally verbose mapping).")
@mcp_tool
def gait_remote_list(verbose: bool = True) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    rems = remote_list(repo)
    return {"ok": True, "remotes": rems if verbose else sorted(list(rems.keys()))}


@mcp.tool(description="Get the URL for a configured remote by name (default: origin).")
@mcp_tool
def gait_remote_get(name: str = "origin") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    url = remote_get(repo, name)
    return {"ok": True, "remote": name, "url": url}


@mcp.tool(description="Create a remote repo on a GAITHUB-compatible server for owner/repo_name.")
@mcp_tool
def gait_repo_create(remote: str, owner: str, repo_name: str, token: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _require_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)
    remote_create_repo(spec, token=tok)
    return {"ok": True, "created": f"{owner}/{repo_name}", "remote": remote}


@mcp.tool(description="Push the current (or specified) branch to a GAITHUB-compatible remote repo.")
@mcp_tool
def gait_push(remote: str, owner: str, repo_name: str, branch: str = "", token: str = "") -> Dict[str, Any]:
    """Push the (current or specified) branch to the named remote; will create remote repo if needed."""
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _require_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)

    try:
        remote_push(repo, spec, token=tok, branch=branch or None)
    except RuntimeError as e:
        msg = str(e)
        if ("Repo not initialized for this owner" in msg) or ("Repo not initialized" in msg):
            remote_create_repo(spec, token=tok)
            remote_push(repo, spec, token=tok, branch=branch or None)
        else:
            raise

    return {"ok": True, "pushed": branch or repo.current_branch(), "remote": remote, "owner": owner, "repo": repo_name}


@mcp.tool(description="Fetch remote heads and memory refs from a GAITHUB-compatible remote repo.")
@mcp_tool
def gait_fetch(remote: str, owner: str, repo_name: str, token: str = "") -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _get_gaithub_token(token)  # allow anonymous if your server supports it
    spec = _remote_spec(repo, remote, owner, repo_name)
    heads, mems = remote_fetch(repo, spec, token=tok)
    return {"ok": True, "remote": remote, "owner": owner, "repo": repo_name, "heads": len(heads), "memory": len(mems)}


@mcp.tool(description="Pull a branch from a GAITHUB-compatible remote and merge into current branch. Optionally merge memory.")
@mcp_tool
def gait_pull(
    remote: str,
    owner: str,
    repo_name: str,
    branch: str = "",
    with_memory: bool = False,
    token: str = "",
) -> Dict[str, Any]:
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    tok = _get_gaithub_token(token)
    spec = _remote_spec(repo, remote, owner, repo_name)

    merge_id = remote_pull(
        repo,
        spec,
        token=tok,
        branch=branch or repo.current_branch(),
        with_memory=with_memory,
    )

    out: Dict[str, Any] = {
        "ok": True,
        "pulled": f"{remote}/{branch or repo.current_branch()}",
        "into": repo.current_branch(),
        "head": merge_id,
    }
    if with_memory:
        out["memory"] = repo.read_memory_ref(repo.current_branch())
    return out


@mcp.tool(description="Clone a GAIT repo from a GAITHUB-compatible remote URL into a local folder.")
@mcp_tool
def gait_clone(
    url: str,
    owner: str,
    repo_name: str,
    path: str,
    remote: str = "origin",
    branch: str = "main",
    token: str = "",
) -> Dict[str, Any]:
    tok = _get_gaithub_token(token)  # may be None if your server allows anonymous clone
    dest = Path(path).expanduser().resolve()

    spec = RemoteSpec(base_url=url, owner=owner, repo=repo_name, name=remote)
    clone_into(dest, spec, token=tok, branch=branch)

    return {
        "ok": True,
        "cloned": f"{owner}/{repo_name}",
        "into": str(dest),
        "branch": branch,
        "remote": remote,
        "url": url,
    }


# ---------------------------------------------------------------------
# AI Context Recovery Tools
# ---------------------------------------------------------------------
@mcp.tool(description="Sync AI state with GAIT history. Call this after a revert or to recover context.")
@mcp_tool
def gait_resume(
    target: str = "HEAD",
    turns: int = 10,
    include_pinned_memory: bool = True,
) -> Dict[str, Any]:
    """
    Rebuilds the AI's ground truth from GAIT history.
    Includes explicit instructions for Copilot to avoid 'ghost turns'.
    """
    repo, err = _try_repo()
    if err:
        return err
    assert repo is not None

    head = repo.head_commit_id() or ""
    if not head:
        return _err("No commits yet; nothing to resume.")

    t = (target or "").strip()
    cid = head if t in ("", "HEAD", "@") else _resolve_commit_prefix_from_head(repo, head, t)

    want = max(0, int(turns))
    collected: List[Dict[str, Any]] = []
    seen = set()
    cur = cid

    # Walk backward through history
    while cur and cur not in seen and len(collected) < want:
        seen.add(cur)
        c = repo.get_commit(cur)
        turn_ids = c.get("turn_ids") or []

        for tid in turn_ids:
            if len(collected) >= want:
                break
            tdata = repo.get_turn(tid)
            
            # Extract text and artifacts from the stored turn
            user_txt = (tdata.get("user") or {}).get("text", "")
            asst_txt = (tdata.get("assistant") or {}).get("text", "")
            artifacts = tdata.get("context", {}).get("artifacts", [])
            
            if user_txt or asst_txt:
                collected.append({
                    "user": user_txt, 
                    "assistant": asst_txt,
                    "artifacts": artifacts
                })

        parents = c.get("parents") or []
        cur = parents[0] if parents else ""

    collected.reverse()

    # The bundle now contains a 'system_instruction' to guide Copilot
    bundle: Dict[str, Any] = {
        "ok": True,
        "current_state": {
            "branch": repo.current_branch(),
            "head": short_oid(cid),
            "total_turns_restored": len(collected)
        },
        "history": collected,
        "instructions": (
            "CRITICAL: History has been synchronized. "
            "1. DO NOT call gait_record_turn for this resume/revert action. "
            "2. Use the provided 'history' as your new current context. "
            "3. If artifacts are present in the history, consider those files to be in their current state."
        )
    }

    if include_pinned_memory:
        try:
            bundle["pinned_memory"] = repo.build_context_bundle(full=False)
        except Exception as e:
            bundle["pinned_memory_error"] = str(e)

    return bundle

if __name__ == "__main__":
    log.info("GAIT MCP Server starting up...")
    try:
        mcp.run()
    except Exception as e:
        log.error(f"Server crashed: {e}")
