# CI Gatekeeper Skill — Design Document

**Date:** 2026-06-09
**Status:** Approved
**Author:** OpenCode
**Skill Name:** `ci-gatekeeper`
**Skill Path:** `~/.config/opencode/skills/ci-gatekeeper/`

---

## 1. Purpose

The **CI Gatekeeper** skill is an OpenCode skill that auto-detects a project's CI requirements and warns about common CI-breaking issues **before** the user commits. It prevents failures like:
- Cross-module private access violations (e.g., `module._private_func()`)
- Missing BUILD/TARGETS/test targets for new files
- Copyright header omissions
- Import ordering/style violations
- Copybara compatibility issues

The skill auto-detects the organization (Google, Meta, NVIDIA, Anthropic/OpenAI) from project structure and applies the appropriate ruleset.

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   File Edit     │────▶│   OrgDetector   │────▶│   RuleEngine    │
│   (Save/Edit)   │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                              ▼                         ▼                         ▼
                    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
                    │  StaticAnalyzer │     │ PatternMatcher  │     │   Reporter      │
                    │   (AST/Regex)   │     │   (Regex/Heur)  │     │ (Inline Output) │
                    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 2.2 Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **OrgDetector** | Auto-detect org from project structure | Scans root for `BUILD`/`WORKSPACE`/`MODULE.bazel` (Google), `TARGETS`/`.buckconfig` (Meta), `setup.py` + `src/` (Anthropic/OpenAI), `CMakeLists.txt` + `.cu` (NVIDIA) |
| **RuleEngine** | Loads and applies rules | YAML-based rulesets per org + common rules |
| **StaticAnalyzer** | Structural analysis | Python `ast` module for cross-module access, BUILD file parsing for target completeness |
| **PatternMatcher** | Style/formatting checks | Regex for copyright headers, import ordering, trailing whitespace |
| **Reporter** | Inline warnings | Formats `[CI Gatekeeper] SEVERITY: message` with context and fix suggestions |

---

## 3. Rules & Checks

### 3.1 Common Rules (All Orgs)

| # | Rule | Engine | Severity | Description |
|---|------|--------|----------|-------------|
| 1 | Cross-Module Private Access | Static Analysis | **ERROR** | Detects `module._private_func()` or `module._PrivateClass()` calls across module boundaries |
| 2 | Missing Test Targets | Static Analysis | **ERROR** | Detects `*_test.py` files without corresponding `py_test`/`cc_test` targets in BUILD/TARGETS |
| 3 | Unused Dependencies | Static Analysis | **WARN** | Detects imports in source files not listed in BUILD file `deps` |
| 4 | Copyright Header | Pattern | **WARN** | Checks all `.py`, `.cc`, `.h` files have the org's standard copyright header |
| 5 | Import Ordering | Pattern | **INFO** | Standard Python import ordering (stdlib → third-party → local) |

### 3.2 Google-Specific Rules

| # | Rule | Engine | Severity | Description |
|---|------|--------|----------|-------------|
| 6 | Bazel BUILD Completeness | Static Analysis | **ERROR** | New `.py` files must be in `py_library`/`py_binary`; new `.cc` files in `cc_library`/`cc_binary` |
| 7 | Copybara Compatibility | Pattern | **WARN** | Checks for `__pycache__` in commits, absolute paths that won't survive copybara migration |
| 8 | Proto/GRPC Patterns | Static Analysis | **WARN** | Detects proto imports that might break in monorepo context |

### 3.3 Meta-Specific Rules

| # | Rule | Engine | Severity | Description |
|---|------|--------|----------|-------------|
| 9 | Buck TARGETS Completeness | Static Analysis | **ERROR** | New files must be listed in appropriate Buck targets |
| 10 | Internal Import Leak | Pattern | **ERROR** | Detects `fb` or `meta` internal import paths that might leak to open source |

### 3.4 Anthropic/OpenAI-Specific Rules

| # | Rule | Engine | Severity | Description |
|---|------|--------|----------|-------------|
| 11 | Python Package Completeness | Static Analysis | **WARN** | New `.py` files must be in `setup.py` or `pyproject.toml` packages |
| 12 | PyTorch/JAX Patterns | Pattern | **WARN** | Checks for common CUDA/device mismatches |

### 3.5 NVIDIA-Specific Rules

| # | Rule | Engine | Severity | Description |
|---|------|--------|----------|-------------|
| 13 | CUDA Build Patterns | Pattern | **WARN** | Checks for nvcc flag consistency, CUDA version compatibility |

---

## 4. File Structure

```
~/.config/opencode/skills/ci-gatekeeper/
├── SKILL.md                    # Main skill definition
├── rules/
│   ├── common.yaml             # Common rules (all orgs)
│   ├── google.yaml             # Google/DeepMind-specific rules
│   ├── meta.yaml               # Meta-specific rules
│   ├── nvidia.yaml             # NVIDIA-specific rules
│   └── anthropic_openai.yaml   # Anthropic/OpenAI rules
├── detectors/
│   ├── org_detector.py         # Auto-detects org from project structure
│   ├── build_file_parser.py    # Parses BUILD/TARGETS/setup.py files
│   └── ast_analyzer.py         # Python AST analysis for cross-module access
└── reporters/
    └── inline_reporter.py      # Formats warnings for OpenCode display
```

---

## 5. Org Detection Logic

```python
def detect_org(project_root):
    files = os.listdir(project_root)
    
    if any(f in files for f in ['BUILD', 'WORKSPACE', 'MODULE.bazel', '.bazelrc']):
        return 'google'
    elif any(f in files for f in ['TARGETS', '.buckconfig', 'BUCK']):
        return 'meta'
    elif 'setup.py' in files and 'requirements.txt' in files:
        return 'anthropic_openai'
    elif any(f.endswith('.cu') for f in files) or 'CMakeLists.txt' in files:
        return 'nvidia'
    else:
        return 'common'
```

---

## 6. Severity Levels

- **ERROR**: Will definitely break CI. Must fix before commit. (e.g., cross-module access, missing test targets)
- **WARN**: Likely to cause issues. Should fix. (e.g., copyright header, copybara compatibility)
- **INFO**: Style suggestion. Optional. (e.g., import ordering)

---

## 7. Example Output

When editing `betti_defense.py`:

```
[CI Gatekeeper] ERROR: Cross-module private access detected at line 128
  → `betti_defense._connected_components()` is called from `global_graph_topology.py`
  → Fix: Rename to `connected_components()` or make it public

[CI Gatekeeper] WARN: Missing copyright header in `betti_defense_test.py`
```

---

## 8. Integration with OpenCode

- **Auto-activates** when editing `.py`, `.bzl`, `.cc`, `.h`, `BUILD`, `TARGETS` files
- **Manual check**: `check CI` or `run ci-gatekeeper`
- **Respects `.gitignore`** and skips `__pycache__`, `.git/`, etc.

---

## 9. Future Extensions (Phase 2)

1. **LSP Integration**: Use symbol-level analysis via rust-analyzer or pylsp for more precise checks
2. **Custom Rules**: Support `.ci-gatekeeper.yaml` in project root for per-project overrides
3. **Auto-Fix**: Suggest fixes (e.g., "Add this to BUILD file: `py_test(...)`")
4. **CI Integration**: Parse actual CI config files (`.github/workflows/`, `.gitlab-ci.yml`) to validate against them

---

## 10. Testing Strategy

1. **Unit Tests**: Each detector and analyzer gets unit tests
2. **Integration Tests**: Test against real open-source projects (e.g., TensorFlow for Google patterns, PyTorch for Meta patterns)
3. **Regression Tests**: Ensure the exact cross-module access issue we fixed (`betti_defense._connected_components`) would be caught

---

## 11. Approval

**User Approval:** ✅ Approved via conversational design review (2026-06-09)

**Sections Approved:**
- Section 1: Architecture & Overview ✅
- Section 2: Rules & Checks ✅
- Section 3: Implementation Details ✅

---

*End of Design Document*
