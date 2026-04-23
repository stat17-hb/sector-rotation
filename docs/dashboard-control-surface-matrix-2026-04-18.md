# Dashboard Control-To-Surface Matrix

Immediate-pass artifact for the approved dashboard-efficiency change.

## Scope
- Current-state documentation only.
- Used to prevent accidental scope widening while removing summary-tab duplication.

## Matrix

```text
Control / Surface                 | Current location                  | Affects today                                            | Does not affect today                                      | Future widened-scope target
----------------------------------|-----------------------------------|----------------------------------------------------------|------------------------------------------------------------|----------------------------
Page header pills                 | app.py page shell                 | Visual status framing only                               | Session state, boards, command bar, canvas, tabs           | Keep as framing-only unless shell-level global mode is added
Top status banner                 | app.py after page header          | Data-source warning/notice framing                       | Session state, boards, command bar, canvas, tab filtering  | Keep as framing-only unless operational controls are introduced
Analysis toolbar                  | app.py before decision stack      | Date range, selected phase/sector/month analysis context | Held/new decision boards, lower command-bar filter state   | Root control layer for shared research-state ownership
Held-sector selection             | render_investor_decision_boards() | Held/new recommendation wording and position-mode output | Header framing, lower command bar, analysis canvas, tabs   | May remain local unless portfolio-wide mode is introduced
Lower command-bar filters         | lower research layer before tabs  | Downstream `signals_filtered` and `top_pick_signals`     | Held/new boards, page header, top banner, analysis canvas  | Keep scoped to the lower research layer unless the research/execution model is redesigned
Inline investor-flow summary      | decision-first stack              | Top-of-page reference context for KR flow                | Tab filtering, boards logic, command-bar scope             | Keep as compact context layer separate from raw/detail tab
Summary tab surfaces              | first tab                         | Top picks and action distribution after current filters  | Hero/status framing after dedupe, board state ownership    | Stay additive; do not become a second landing page
All-signals / chart / flow tabs   | downstream tabs                   | Detailed inspection using downstream filtered views      | Top-of-page framing and earlier board rendering            | Potential consumers of a future unified control model
```

## Current Reading
- The most important mismatch today is not block order alone.
- The lower command bar belongs to the lower research layer.
- It governs the downstream filtered views that consume `signals_filtered` and `top_pick_signals`.
- The analysis toolbar still owns research scope, while held/new boards keep their own owner through `held_sectors`.

## Immediate Constraint
- This pass keeps distinct owners instead of falsely page-global filters:
  - analysis toolbar -> research scope
  - held-sector selection -> execution context
  - lower command bar -> downstream research filtering
- It does not merge those domains into one toolbar.

## Follow-Up Gate
- Do not widen the lower command bar upward unless the research/execution state model is intentionally redesigned.
