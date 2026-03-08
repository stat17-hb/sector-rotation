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
- Pattern: `pre-commit` local hook used `language: system` with `entry: python ...`, causing Windows commit failures (`exit code 9009`) when commit-time PATH lacked `python`.
- Rule: For Python-based local pre-commit hooks, prefer `language: python` to avoid PATH-dependent failures across IDE/CLI commit environments.
- Pattern: User reported `run_streamlit.bat` showed no output after simplifying launcher commands.
- Rule: In Windows batch scripts, keep `call` when invoking `conda activate` (batch-to-batch); otherwise control flow may not return to run the Streamlit command.
- Rule: For local launcher scripts, preserve `%*` argument forwarding and include explicit activation-failure messages so double-click runs do not fail silently.
- Pattern: `conda activate` failed in CMD with `Run 'conda init' before 'conda activate'` even though conda was installed.
- Rule: For CMD launchers, prefer resolving `conda.bat` (via `CONDA_EXE` or known `condabin` paths) and call it directly; do not assume `conda init cmd.exe` was run.

## 2026-02-24
- Pattern: Light theme looked low-contrast despite acceptable token-level WCAG checks.
- Rule: For theme toggles, verify readability at component level (native Streamlit widgets, sidebar controls, markdown body text), not only token contrast pairs.
- Rule: Avoid applying `text_muted` to global markdown paragraphs; reserve muted color for captions/labels and keep body copy at base text color.
- Pattern: Light background + dark dataframe default skin created visual mismatch and poor readability.
- Rule: For Streamlit tables, apply theme at component level (`pandas.Styler`) first and keep CSS selectors only as fallback; do not rely on root CSS tokens alone.
- Pattern: Streamlit top chrome (`stHeader`/`stDecoration`) and Glide Data Grid can ignore app background tokens and remain dark.
- Rule: In light mode, explicitly style `stHeader`, `stDecoration`, and Glide Data Grid CSS variables; token updates alone are insufficient.
- Pattern: FX shock behavior was documented but runtime signal path still passed `fx_change_pct=0.0`, disabling the alert in practice.
- Rule: For risk/input wiring changes, validate end-to-end propagation in integration tests (`caller -> build_signal_table -> scoring`) instead of relying only on helper unit tests.
- Rule: Do not hardcode neutral sentinel values (`0.0`) for optional market inputs; pass computed runtime values and use `NaN` when data is unavailable.

## 2026-03-02
- Pattern: `app.py` UI 문자열이 모지바케로 저장되어 한글이 깨지고, `st.warning(..., icon=...)`에 깨진 문자열이 들어가 `invalid emoji` 예외가 발생함.
- Rule: UI 텍스트/주석이 많은 파일은 수정 후 `python -m py_compile <file>`와 `rg "�|\\?곗|\\?꾩|\\?좏"` 같은 깨짐 패턴 점검을 최소 1회 실행한다.
- Rule: Streamlit `icon=` 인자는 항상 단일 유효 이모지(`⚠️`, `✅` 등)만 사용하고, 일반 문자열/깨진 문자열은 절대 넣지 않는다.
- Rule: VS Code/IDE 및 저장 설정을 UTF-8(권장: UTF-8 without BOM)으로 고정하고, 인코딩 전환 저장(CP949/EUC-KR)을 금지한다.

## 2026-03-08
- Pattern: User reported that the cycle timeline visually collapsed into near-invisible lines, so regime colors were technically present in code but not legible in the actual chart.
- Rule: For time-interval visuals, verify that the rendering primitive matches the semantics of a duration; do not use point-like bars for date spans when the goal is a visible band/timeline.
- Rule: When a user calls out “can’t distinguish periods,” inspect both palette contrast and chart geometry before treating it as a theme-only bug.
- Pattern: A real-network `bootstrap_warehouse.py` run failed immediately with `_duckdb.IOException` because a local `streamlit run app.py` process still held a write lock on `data/warehouse.duckdb`.
- Rule: Before any warehouse bootstrap/backfill/sync that writes to DuckDB, check for repo-local `python -m streamlit run app.py` or other writer processes and stop them first; otherwise treat file-lock errors as environment/process conflicts before changing code.
- Pattern: `git push` to GitHub failed because a regenerated `data/warehouse.duckdb` blob in the latest commit exceeded the 100 MB file limit.
- Rule: Before committing or pushing, inspect staged/generated artifacts for size-sensitive paths (`data/`, `backups/`, `logs/`) and keep local DuckDB files untracked via `.gitignore` plus `git rm --cached` when they were already tracked.
