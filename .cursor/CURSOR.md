# Cursor AI agent configuration

## Overview

This project uses **Cursor** with the same structured workflows, specialized agent personas, and coding standards as the Claude-oriented **`.claude/`** tree. Cursor-specific files live under **`.cursor/`**.

---

## Development workflow

Follow this workflow for feature development:

```
/spec  →  /plan  →  /build  →  /test  →  /review  →  Ship
```

| Phase | Prompt source | Purpose |
|-------|----------------|--------|
| **Define** | `.cursor/commands/spec.md` | PRD: objectives, scope, boundaries |
| **Plan** | `.cursor/commands/plan.md` | Vertical slices, acceptance criteria |
| **Build** | `.cursor/commands/build.md` | Incremental implementation, TDD |
| **Verify** | `.cursor/commands/test.md` | Tests and verification |
| **Review** | `.cursor/commands/review.md` | Five-axis review before merge |
| **Ship** | `.cursor/commands/deploy.md` | Build, test, deploy |

### Supporting prompts

| File | Purpose |
|------|---------|
| `commands/debug.md` | Systematic diagnosis |
| `commands/simplify.md` | Reduce complexity, same behavior |
| `commands/fix-issue.md` | Analyze and fix reported issues |

**How to use:** Open the markdown file, copy the section you need, or **@ mention** the file in Chat/Composer so the model loads it.

---

## Core principles

- **TDD** — Failing tests first, then implementation (`.cursor/skills/tdd/`)
- **Incremental implementation** — Small vertical slices (`.cursor/skills/incremental-implementation/`)
- **Five-axis review** — Correctness, readability, architecture, security, performance (`.cursor/skills/code-review/`)

---

## Mandatory rules (Cursor)

Project rules are **`.cursor/rules/*.mdc`**. They use YAML frontmatter:

- **`alwaysApply: true`** — Included in every relevant session (`cursor-overview.mdc`, `security.mdc`)
- **`alwaysApply: false`** + **`globs`** — Applied when files matching the pattern are in context

| Topic | Rule file |
|-------|-----------|
| Clean code, style, errors | `clean-code`, `code-style`, `error-handling` |
| Stack, structure, APIs | `tech-stack`, `project-structure`, `api-conventions` |
| Data & naming | `naming-conventions`, `database` |
| Ops & quality | `security`, `monitoring`, `testing`, `git-workflow`, `system-design` |

---

## Agent personas

Instructions live in **`.cursor/agents/`**. Invoke by **@ mentioning** the file (e.g. `@.cursor/agents/backend.md`).

| Area | File |
|------|------|
| Frontend, backend, architecture | `frontend.md`, `backend.md`, `systems-architect.md` |
| Quality | `code-reviewer.md`, `test-engineer.md`, `qa.md`, `security-auditor.md` |
| Product & content | `project-manager.md`, `ui-ux-designer.md`, `copywriter-seo.md` |

---

## Skills

Reusable playbooks: **`.cursor/skills/*/SKILL.md`** (and related `.md` files where present).

| Skill | Use for |
|-------|---------|
| `tdd` | Red–green–refactor |
| `code-review` | Five-axis review |
| `incremental-implementation` | Vertical slices |
| `deploy` | Deployment pipeline |
| `security-review` | Security audit |

---

## Reference checklists

**`.cursor/references/`**

| File | Use for |
|------|---------|
| `security-checklist.md` | Pre-deploy security |
| `testing-patterns.md` | Test structure |
| `performance-checklist.md` | Performance |
| `accessibility-checklist.md` | WCAG-oriented checks |

---

## Config parity

**`.cursor/settings.json`** lists directories (mirrors `.claude/settings.json` for Claude Code). Cursor natively loads **`.cursor/rules/*.mdc`**; other paths are documentation for humans and for `@` includes.

---

## Agent behavior

1. Follow the workflow and use the command prompts when starting a phase.
2. Apply **`.cursor/rules/`**; treat **`security.mdc`** as non-negotiable.
3. Prefer tests first and small, buildable changes.
4. **@ mention** the right **`.cursor/agents/`** file when the task matches that role.
