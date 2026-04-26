# Frontend Fundamentals Design Contract

This project follows Toss Frontend Fundamentals as the primary frontend design and code-quality standard.

References:

- https://github.com/toss/frontend-fundamentals
- https://frontend-fundamentals.com/code-quality/en/code/

The governing goal is simple: frontend code and UI structure must be easy to understand, easy to change, and safe to verify. Visual polish matters, but it is subordinate to modifiability, accessibility, predictable behavior, and truthful financial information design.

## Core Standard

Toss Frontend Fundamentals defines good frontend code as code that is easy to modify and deploy when requirements change. This project translates that into four non-negotiable criteria:

- Readability: a maintainer can understand the screen, data flow, and rendering logic without holding many contexts at once.
- Predictability: a component, function, state key, CSS class, or test behaves as its name and contract imply.
- Cohesion: things that change together are located, named, and verified together.
- Coupling: a change has a narrow and visible impact radius.

When these criteria conflict, choose based on future modification risk:

- If missing one coordinated update can break behavior, prioritize cohesion.
- If abstraction only hides simple local behavior, prioritize readability.
- If sharing code would create broad impact for unrelated screens, allow duplication.
- If duplication would let product rules drift, centralize the rule and test it.

## Product Scope

The app is a Streamlit sector-rotation dashboard for investment research support.

It may present:

- market and regime context;
- sector and stock lookup;
- model output categories;
- held/new decision-board state;
- investor-flow context;
- returns, volatility, and MDD;
- alerts and validation context;
- ranking-like research views only when backed by existing data.

It must not imply:

- brokerage account access;
- order entry;
- live order execution;
- account holdings from a brokerage;
- live transaction-value or live volume rankings unless a concrete data contract exists;
- personalized suitability advice;
- guaranteed outcomes.

Toss Invest remains a product-rhythm reference only: compact navigation, visible search, filterable discovery, ranking-like research surfaces, investor-flow context, Korean-readable copy, and clear disclaimers. It is not a pixel-clone target and does not override the four code-quality principles above.

## Readability

Readable frontend code reduces the number of contexts a maintainer must track.

### Structure

- Separate code that does not run together.
- Keep data loading, state normalization, rendering, styling, and copy rules in distinct layers.
- Prefer top-down file flow: public entry point first, helpers below, constants near the behavior they explain.
- Avoid mixed-purpose functions that validate data, mutate state, build CSS, and render UI in one block.
- Split functions by logic type when a maintainer would otherwise need to reason about unrelated branches.

Expected boundaries:

- `app.py`: page orchestration, route selection, high-level composition.
- `src/dashboard/*`: dashboard tab contracts and screen-level flow.
- `src/ui/*`: reusable renderers, panels, tables, figures, CSS, copy, and visual primitives.
- `config/theme.py`: shared theme tokens and semantic colors.
- `.streamlit/config.toml`: Streamlit runtime theme bridge.
- `tests/*`: behavior, copy, theme, contrast, component, and runtime verification.

### Naming

- Name complex conditions before using them in UI branches.
- Name magic numbers as domain or layout constants.
- Prefer semantic names over visual implementation names.
- Do not encode stale inspiration sources in names unless the source is still the active product contract.
- Avoid names that imply unsupported financial authority, such as order, trade, execution, account, holding, or suitability.

Good naming examples:

- `has_supported_ranking_data`
- `render_sector_lookup_panel`
- `DECISION_SUPPORT_COPY`
- `REGIME_STATUS_COLORS`

Poor naming examples:

- `do_stuff`
- `new_ui`
- `toss_clone_card`
- `buy_button`
- `magic_blue`

### Copy

Copy is part of readability.

- Use compact Korean labels where they reduce scan cost.
- Use English only where it is already a known financial abbreviation or improves precision.
- Keep disclaimers close to the claim they qualify.
- Reframe imperative buy/sell language as candidate, review, signal, or risk context unless it is explicitly a model category label.
- Explain unavailable or empty states in the user's task language, not implementation language.

Allowed terms:

- decision support;
- candidate;
- rules-based;
- research validation;
- signal context;
- watch;
- review;
- risk context.

Avoid:

- direct trade commands;
- unsupported real-time claims;
- copy that implies brokerage authority;
- vague decorative labels that hide the actual data source.

## Predictability

Predictable frontend code behaves consistently from its public contract.

### Components and Functions

- Similar render functions must accept similar input shapes and return similar output shapes.
- Functions named `get_*` or `build_*` should not mutate Streamlit session state.
- Functions named `render_*` may render UI but should not fetch remote data unless their name and docstring make that side effect explicit.
- State normalization should happen before rendering.
- Empty, loading, disabled, and error states must be explicit branches, not incidental fallthrough behavior.

### State and Routing

- Route keys, tab IDs, session-state keys, and CSS hooks must have one canonical spelling.
- User-visible navigation labels may change, but internal IDs should stay stable unless tests and migration points are updated together.
- Hidden behavior must be surfaced through names, constants, comments, or tests.
- If a screen has a default selection, the default must be defined in one place and tested.

### Visual Behavior

- Interactive controls must have visible hover, selected, focus, disabled, empty, and error states where Streamlit allows them.
- A selected filter or tab must be visually distinguishable without relying only on color.
- Charts and tables must use stable dimensions so hover states, legends, labels, and loading text do not shift layout unexpectedly.
- Responsive behavior must preserve information hierarchy across desktop and narrow widths.

## Cohesion

Cohesive frontend code keeps things that change together close together.

### Token and Style Cohesion

- Shared colors, spacing, radii, font stacks, and chart semantics belong in theme/CSS token surfaces.
- Financial positive, negative, neutral, and warning colors must be defined semantically, not re-created ad hoc.
- Repeated panel, table, metric, and status patterns should use existing UI helpers before new markup is introduced.
- CSS selectors should attach to stable semantic classes, not fragile generated DOM structure, unless Streamlit leaves no alternative.

### Product Rule Cohesion

- Product boundaries, forbidden claims, copy terms, and disclaimer rules must be changed together.
- If a new UI surface introduces a financial claim, update the copy tests or runtime tests that protect that claim.
- Ranking-like UI must be backed by the same data-contract documentation and tests that define its source.
- Data availability rules and empty-state copy should live close enough that one cannot change without noticing the other.

### Test Cohesion

- Theme changes should update theme and contrast tests together.
- Copy changes should update copy tests together.
- Tab or route changes should update dashboard tab/runtime tests together.
- New reusable UI helpers should get focused tests when they encode product rules, accessibility rules, or financial semantics.

## Coupling

Low-coupling frontend code limits the blast radius of a change.

### Responsibility Boundaries

- Keep orchestration, data transformation, presentation, and styling responsibilities separate.
- Do not let one renderer know about every tab unless it is the screen-level coordinator.
- Do not push unrelated data through long parameter chains just because a distant child might need it.
- Avoid global CSS changes unless the selector is deliberately scoped and verified against all affected pages.
- Avoid session-state writes in low-level UI helpers unless state ownership is part of that helper's explicit contract.

### Duplication Policy

Duplication is allowed when it lowers coupling and keeps behavior easier to inspect.

Allow local duplication for:

- small layout variations;
- one-off Streamlit workarounds;
- screen-specific copy;
- temporary visual experiments that are not yet product rules.

Centralize instead when:

- the duplicated value is a product rule;
- the duplicated value is a financial semantic;
- the duplicated code controls accessibility;
- the duplicated behavior appears across multiple routes and must evolve together.

### Change Radius

Before modifying shared UI or CSS, identify affected screens.

Expected safe change radii:

- Copy-only change: `src/ui/copy.py` plus copy tests.
- Theme-token change: `config/theme.py`, CSS bridge, theme/contrast tests.
- Component behavior change: component module plus focused UI tests.
- Route/tab change: dashboard routing plus runtime/tab tests.
- Product-boundary change: `DESIGN.md`, copy layer, and tests that protect unsupported claims.

## Visual System

The visual system must support readability and predictable scanning before decoration.

### Atmosphere

- Quiet, app-like, information-dense.
- Light mode is the authority; dark mode is an accessible companion.
- No decorative gradients, bokeh, oversized hero marketing, or generic blue-purple AI styling.
- Depth comes from surface contrast, hairline borders, and subtle shadows.

### Color

- Use blue as a restrained interaction and selection accent.
- Use semantic colors for positive, negative, neutral, warning, and disabled states.
- Keep text contrast accessible in both light and dark modes.
- Do not encode meaning only through color.
- Tune chart colors for table/chart readability before brand expression.

### Typography

- Korean readability wins over display drama.
- Use Pretendard, Noto Sans KR, or UI sans stacks.
- Use tabular numeric rendering for financial values.
- Reserve large type for page identity only.
- Do not scale font size with viewport width.
- Keep letter spacing neutral except for small labels.

### Geometry and Layout

- Cards and table containers should stay compact.
- Controls should be compact rounded rectangles, not oversized marketing CTAs.
- Repeated cards should be scannable and data-led.
- Fixed-format surfaces need stable responsive dimensions.
- First viewport should expose identity, market/regime context, search or lookup, navigation/filter controls, and the primary decision-support entry point.

## Accessibility

Accessibility is not optional polish.

- Keyboard focus must remain visible.
- Interactive controls need discernible text or accessible names.
- Error and empty states must be readable without visual context alone.
- Contrast-sensitive text, chart labels, and table text must be verified when colors change.
- Icon-only controls need labels or tooltips when Streamlit supports them.
- Motion, if introduced, must not be required to understand state.

## Implementation Checklist

Use this checklist before shipping frontend changes:

- Readability: Can a maintainer understand the screen without jumping across unrelated files?
- Readability: Are complex conditions and magic numbers named?
- Predictability: Do function names, inputs, outputs, and side effects match?
- Predictability: Are empty, loading, disabled, selected, and error states explicit?
- Cohesion: Are files that must change together located or tested together?
- Cohesion: Are product rules and copy/disclaimer rules protected by tests?
- Coupling: Is the change radius narrow and visible?
- Coupling: Would local duplication be safer than a shared abstraction?
- Accessibility: Are focus, contrast, keyboard, and semantic text states preserved?
- Finance truth: Does every claim map to an existing data source or documented limitation?

## Verification Contract

Required for Python/UI changes:

- `python -m py_compile` on changed Python files.
- Focused tests for the touched surface.
- For broad UI changes:
  `pytest -q tests/test_ui_theme.py tests/test_ui_contrast.py tests/test_ui_components.py tests/test_ui_copy.py tests/test_dashboard_tabs.py tests/test_dashboard_runtime.py`

Required for documentation-only changes:

- Confirm the document names the active standards.
- Confirm the four principles are mapped to repository-specific implementation rules.
- Confirm no obsolete design authority is presented as the top-level standard.

Optional when local tooling is stable:

- Streamlit runtime smoke for `app.py`.
- Desktop and narrow screenshots.
- Visual comparison against the latest accepted artifact.

Completion requires no known overlap, clipping, unreadable chart/table text, hidden disclaimers, unsupported brokerage claims, or untested broad frontend behavior changes.
