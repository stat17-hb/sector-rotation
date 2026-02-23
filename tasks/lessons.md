# Lessons Learned

## 2026-02-22
- Pattern: User asked to reconstruct a high-level evaluation into actionable execution.
- Rule: When feedback asks for "restructure", produce a concrete phased checklist with gates, verification commands, and review template in `tasks/todo.md`.
- Rule: If task-management conventions exist in `AGENTS.md`, persist plan artifacts in `tasks/` instead of only replying in chat.
- Pattern: Source-of-truth plan (`plan.md`) was revised and task checklist drifted.
- Rule: Before implementation, diff `plan.md` vs `tasks/todo.md` on contracts, cache behavior, return types, and test matrix; then sync `todo.md` to the latest plan.
- Pattern: User confirmed IDE view is normal while CLI output looked garbled.
- Rule: Treat mojibake seen in constrained PowerShell output as a display-encoding issue first; do not claim file corruption unless byte-level or editor-level evidence confirms it.

## 2026-02-23
- Pattern: User explicitly requested implementation of an already-agreed plan.
- Rule: When the user says "implement this plan", start coding immediately; do not re-propose alternatives unless a blocker is discovered.
- Rule: For approved plans, record execution checklist + verification evidence in `tasks/todo.md` as part of implementation (not as a follow-up suggestion).
