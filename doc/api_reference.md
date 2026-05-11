# API Reference Plan

The project does not yet generate API reference documentation automatically.

## Recommended First Step

Add a root `Doxyfile` and generate HTML directly with Doxygen:

```bash
doxygen Doxyfile
```

Suggested output:

```text
doc/api/html/
```

## Future Sphinx Setup

For a polished documentation site, use:

```text
Sphinx + Breathe + Exhale + Doxygen
```

This would allow hand-written guides and generated C++ API pages to coexist.

## Doxygen Coverage Priorities

Start documenting:

- `StructuredSolver`
- `Solution`
- `StructuredElementContainer`
- equation classes
- boundary condition classes
- time integrators

Focus comments on template parameters, ownership, device/host expectations, and
valid combinations of components.

