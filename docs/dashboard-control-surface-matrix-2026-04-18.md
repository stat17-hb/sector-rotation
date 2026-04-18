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
Analysis toolbar                  | app.py before decision stack      | Date range, selected phase/sector/month analysis context | Held/new decision boards, lower command-bar filter state   | Candidate root control layer for unified research-state ownership
Held-sector selection             | render_investor_decision_boards() | Held/new recommendation wording and position-mode output | Header framing, lower command bar, analysis canvas, tabs   | May remain local unless portfolio-wide mode is introduced
Lower command-bar filters         | render_top_bar_filters()          | Downstream `signals_filtered` and `top_pick_signals`     | Held/new boards, page header, top banner, analysis canvas  | Only move upward if widened to govern boards/canvas/tables consistently
Inline investor-flow summary      | decision-first stack              | Top-of-page reference context for KR flow                | Tab filtering, boards logic, command-bar scope             | Keep as compact context layer separate from raw/detail tab
Summary tab surfaces              | first tab                         | Top picks and action distribution after current filters  | Hero/status framing after dedupe, board state ownership    | Stay additive; do not become a second landing page
All-signals / chart / flow tabs   | downstream tabs                   | Detailed inspection using downstream filtered views      | Top-of-page framing and earlier board rendering            | Potential consumers of a future unified control model
```

## Current Reading
- The most important mismatch today is not block order alone.
- The lower command bar sits in the main body, but its state only drives downstream filtered views.
- That makes it look broader than it is.

## Immediate Constraint
- This pass removes duplicated hero/status from the summary tab only.
- It does not widen control scope.
- It does not relocate the lower command bar.

## Follow-Up Gate
- Do not move or relabel the lower command bar as page-global until the widened-scope target is implemented in code.
