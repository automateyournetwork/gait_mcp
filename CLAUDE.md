# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Git Workflow Rules (MANDATORY)

**NEVER push directly to `main` or `master` branch. Always follow this workflow:**

1. Create a feature branch from main:
   ```bash
   git checkout main && git pull origin main
   git checkout -b feature/<descriptive-name>
   ```

2. Make commits to the feature branch

3. Push the feature branch:
   ```bash
   git push -u origin feature/<descriptive-name>
   ```

4. Create a Pull Request and assign to `rhuanssauro`:
   ```bash
   gh pr create --title "type: Description" --body "..." --base main --assignee rhuanssauro
   ```

5. **Wait for review and merge approval** - never merge without approval

**Branch naming conventions:**
- `feature/<name>` - New features
- `fix/<name>` - Bug fixes
- `docs/<name>` - Documentation updates
- `chore/<name>` - Maintenance tasks

**Commit message prefixes:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `chore:` - Maintenance
- `refactor:` - Code refactoring

---

