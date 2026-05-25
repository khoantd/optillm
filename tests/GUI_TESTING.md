# GUI and approach testing guide

Use this guide with the Gradio chat UI (`optillm --gui`) and with `tests/test_cases.json`.

In the UI, open **Testing guide & settings**: the guide table is shown at the top; use **Load sample** to fill approach, system prompt, and **Sample message** (copy that text into the chat box).

While a reply is generating:

1. The **chat** shows a live **Reasoning** block (monospace), then the **Answer**.
2. Open **Routing & reasoning (live)** — the accordion **below the chat** (above Testing guide) for routing metadata and the same reasoning trace.

Works when the model returns thinking tags (e.g. redacted_thinking, thinking, or think). The GUI turns on full-response mode automatically so those tags are preserved (restart with `python optillm.py --gui`).

## Two different goals

| Goal | Approach dropdown | What to paste in chat |
|------|-------------------|------------------------|
| **Exercise a specific technique** | Set the slug (e.g. `moa`, `z3`) | Full problem text from the table below |
| **Exercise the ML auto-router** | `auto` or `predict` | Full problem text only (not category labels) |

**Do not** use meta prompts like `Hard reasoning: prove √2…` or `Short math: GSM8K…`. The router was trained on complete problems, not technique names. Those lines will usually resolve to `none` (confidence below the default 30% threshold).

## Manual testing (recommended)

Pick the **Approach** in the GUI (or use `moa-gpt-4o-mini` / `<optillm_approach>moa</optillm_approach>` in API calls). Paste the **query** (and **system prompt** when listed) from `test_cases.json`.

| Technique | Slug | `test_cases.json` name | System prompt |
|-----------|------|------------------------|---------------|
| Mixture of Agents | `moa` | Arena Bench Hard | (empty) |
| Best of N | `bon` | GSM8K or Simple Math Problem | see case |
| CoT + reflection | `cot_reflection` | GenSelect Math, Reasoning Token Test - Complex Logic (marbles), or Multi-Step Problem | see case |
| MCTS | `mcts` | Reasoning Token Test - Strategic Thinking (Monty-Hall-style doors) | see case |
| PlanSearch | `plansearch` | Maths Problem (LP) or Reasoning Token Test - Algorithm Design | see case |
| ReRead | `re2` | (not in file) use README example: *How many r's are there in strawberry?* | (empty) |
| MARS | `mars` | GH (competition-style complex math) | (empty) |
| Z3 | `z3` | r/LocalLLaMA (potato logic puzzle) | (empty) |
| Self-consistency | `self_consistency` | Reasoning Token Test - Counter-intuitive (two children) | see case |

Copy fields verbatim from JSON:

```json
{
  "name": "Arena Bench Hard",
  "system_prompt": "",
  "query": "Write a Python program to build an RL model..."
}
```

### Example (MoA)

1. **Approach:** `moa`
2. **System prompt:** leave empty
3. **Message:** paste the full `query` from **Arena Bench Hard**

The routing panel may show no ML metadata when the approach is fixed—that is expected.

## Auto / predict mode

With **Approach: `auto`**, the server runs [optillm-modernbert-large](https://huggingface.co/codelion/optillm-modernbert-large) unless disabled (`OPTILLM_AUTO_ROUTER=false`).

Default behavior:

- **`min_confidence` 0.30** (server default): if the top softmax score is below 30%, the resolved approach is **`none`** (plain model call). Logs look like: `confidence 0.177 below min 0.300, using fallback none`.
- The router’s top label is **not guaranteed** to match the “best human technique” for that problem type.

### What to expect on real `test_cases.json` queries

Empirical behavior with default `min_confidence=0.20` and `effort=0.7` (your environment may vary slightly):

| Case | Typical ML top label | Often resolves to |
|------|----------------------|-------------------|
| Arena Bench Hard (RL numpy) | `moa` (~16%) | `none` (fallback) |
| GSM8K | `z3` (~25%) | `none` (fallback) |
| Potato logic | `none` (~19%) | `none` |
| Marbles | `none` (~21%) | `none` (fallback) |
| Monty Hall doors | `leap` (~21%) | `none` (fallback) |
| LP maximize | `leap` (~16%) | `none` (fallback) |
| Algorithm design | `z3` (~41%) | **`z3`** (passes threshold) |

So **auto is conservative**: most benchmark prompts still passthrough. That is by design, not a broken GUI.

### Tuning auto (API / advanced)

Per request:

```python
extra_body={
    "optillm_approach": "auto",
    "optillm_auto_router": {
        "effort": 0.7,
        "min_confidence": 0.15,  # lower → more techniques run; more false positives
        "fallback": "none",
    },
}
```

Lower `min_confidence` applies the ML top label even when uncertain; it does **not** force the label to match the manual table above.

## Routing panel

After each message with `auto` or `predict`, the panel shows:

- **Resolved** — technique actually used
- **confidence** — softmax score for the chosen label
- **source** — `ml`, `fallback`, or `heuristic`
- **Top scores** — runner-up labels

If **Resolved** is `none` and **source** is `fallback`, either raise `min_confidence` tuning (see above) or switch to a **manual** slug for the technique you want to test.

## Running the main suite vs the GUI

| Command | Purpose |
|---------|---------|
| `python tests/test.py --approaches moa` | Automated run against `test_cases.json` |
| `optillm --gui` | Interactive chat; set approach in the accordion |

See [tests/README.md](README.md) for pytest and CI details.

## Quick reference: full prompts to try

**Arena Bench Hard (moa):**

> Write a Python program to build an RL model to recite text from any position that the user provides, using only numpy.

**GSM8K (bon):**

> If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

**Potato logic (z3):**

> I have a dish of potatoes. The following statements are true: No potatoes of mine, that are new, have >been boiled. All my potatoes in this dish are fit to eat. No unboiled potatoes of mine are fit to eat. Are there any new potatoes in this dish?

**Monty-Hall-style (mcts):** use the full door/light query from *Reasoning Token Test - Strategic Thinking* in `test_cases.json`.

More cases: open `tests/test_cases.json` and copy `query` / `system_prompt` for the row you need.
