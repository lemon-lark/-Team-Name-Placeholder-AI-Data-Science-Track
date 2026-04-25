# AI-partments

**Smarter renter intelligence for NYC.**

AI-partments is a Streamlit dashboard that turns natural-language renter questions
into safe DuckDB SQL, charts, maps, and recommendations. It works out of the
box in Mock Demo Mode (no API keys, no external services) and can ingest real
NYC Open Data CSVs to give live renter intelligence over the boroughs and
neighboring NJ neighborhoods.

Ask questions like:

- *Find 1-bedroom apartments under $2,500 in low-crime neighborhoods near parks.*
- *Compare Astoria, Harlem, and Flushing by transit, parks, and housing supply.*
- *Rank neighborhoods by `renter_fit_score`.*
- *Which neighborhoods have the best subway access and affordable rent?*
- *Which neighborhoods have lots of HPD housing stock?*

## Quickstart

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd.exe
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m src.seed_data
streamlit run app.py
```

> **Windows note:** if PowerShell shows `running scripts is disabled on this system`
> when you try to run `Activate.ps1`, run the following once for your user:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
>
> Or skip activation entirely and call the venv's Python directly:
>
> ```powershell
> .\.venv\Scripts\python.exe -m src.seed_data
> .\.venv\Scripts\python.exe -m streamlit run app.py
> ```

The first launch generates `db/renteriq.duckdb` with 12 NYC/NJ neighborhoods
and synthetic apartments, amenities, parks, facilities, transit stops, and
crime events. Open the **Ask AI** tab and click any of the example questions.

## Dashboard tabs

- **Compare neighborhoods** - compare 2-4 neighborhoods across renter-fit metrics, with metric definitions and underlying values.
- **Top neighborhoods** - rank neighborhoods by a selected metric (default: `renter_fit_score`) with clear tie/missing-value handling.
- **Methodology** - explains score formulas, AI safety boundaries, fallback behavior, and data limitations.
- **About** - summarizes architecture and what is deterministic vs AI-assisted.

## UI theme customization

The dashboard now ships with a dark professional visual theme by default.

- Core Streamlit styling tokens (colors, borders, radius, shadows) live in `CUSTOM_CSS` in `app.py`.
- Plotly dark chart defaults are defined in `src/tools/chart_tools.py`.
- Pydeck dark map style, marker colors, and tooltip styling are defined in `src/tools/map_tools.py`.

If you want to tweak the look, update these three locations to keep the app visually consistent.

## Operating modes

### Mock Demo Mode (default)

- No API keys, no Ollama, no IBM credentials needed.
- Synthetic but plausible data is auto-generated on first launch.
- The router/SQL agents use deterministic keyword rules; results always come
  from the local DuckDB database.

### Ollama (local LLM)

```bash
# In a separate terminal
ollama serve
ollama pull llama3.1:8b
# Optional - if you can grab Granite locally:
ollama pull granite3.3:8b
```

Then in the sidebar, switch the LLM provider to **Ollama (local)** and (if
you pulled a different model) update the model name. AI-partments uses the
`ollama` Python package when available and falls back to direct HTTP calls
against `http://localhost:11434/api/chat`. If Ollama is unreachable the agent
pipeline degrades gracefully back to Mock Demo Mode.

### IBM Granite via watsonx.ai

1. Copy `.env.example` to `.env`.
2. Fill in:
   - `WATSONX_API_KEY`
   - `WATSONX_PROJECT_ID`
   - `WATSONX_URL` (e.g. `https://us-south.ml.cloud.ibm.com`)
   - `WATSONX_MODEL_ID` (default: `ibm/granite-3-3-8b-instruct`)
3. Switch the sidebar provider to **IBM Granite / watsonx.ai**.

If credentials are missing the app stays in Mock Demo Mode with a friendly
warning instead of crashing.

## Real NYC data

Drop CSV/XLSX/XLS files into `data/raw/` (or any of the dataset subfolders),
then either run:

```bash
python -m src.ingestion.pipeline
```

or open the **NYC Data Pipeline** tab in the app and click **Profile raw
folder** -> **Ingest all raw files** -> **Rebuild metrics only**.

Folder layout:

```
data/raw/
  apartments/
  demographics/
  crime/
  transit/
  amenities/
  parks/
  worship/
  housing/
  facilities/
  population/
```

Files can also live directly in `data/raw/` - the pipeline detects the
dataset type from the filename.

### Supported NYC datasets

| Filename contains            | Dataset type      |
|------------------------------|-------------------|
| `acs_demo` / `acs`           | demographics      |
| `Buildings_Subject_to_HPD`   | housing           |
| `Facilities_Database`        | facilities        |
| `MTA_Bus_Stops`              | transit_bus       |
| `MTA_Subway_Stations`        | transit_subway    |
| `Population_By_Neighborhood` | population        |
| `Parks_Properties`           | parks             |

### A note on the bus stops file

For very large transit files (the MTA bus stops CSV can have millions of
rows), ingestion may take longer. The pipeline:

1. Streams the file into a DuckDB staging table (`raw_mta_bus_stops`) using
   `read_csv_auto` instead of pandas.
2. Deduplicates rows by `stop_id` (or by `stop_name` + rounded lat/lng if no
   stop id is present) before computing transit metrics.
3. Stores only the unique physical stops in `transit_bus_stops`.

You'll see a warning in the app explaining this whenever a transit-bus file
is profiled.

## Project layout

```
app.py                   Streamlit dashboard (entry point)
requirements.txt         Pinned dependencies
.env.example             LLM provider env var template
README.md                You are here

src/
  config.py              Paths, DB location, LLM env defaults
  schema.py              Canonical table/column definitions
  database.py            DuckDB connection + safe query helpers
  seed_data.py           Demo seed generator (12 NYC/NJ neighborhoods)
  llm_client.py          Mock + Ollama + watsonx providers

  agents/                Router, schema, SQL, safety, visualization, recommendation
  tools/                 Query, chart, map helpers
  ingestion/             CSV/XLSX loader, profiler, cleaner, geospatial,
                         metric engineering, end-to-end pipeline

data/
  raw/                   Drop CSV/XLSX/XLS files here for ingestion
  processed/             CSV outputs from the pipeline
  sample/                Auto-generated demo CSVs

db/
  renteriq.duckdb        Local DuckDB database (filename retained for backward compatibility)

scripts/
  smoke_test.py          Runs all 7 example questions end-to-end
  test_safety.py         Adversarial + happy-path SQL safety cases
  test_clarification.py  Dimension counting, proximity extraction, clarification
  evaluate_pipeline.py   Static agentic evaluation with JSON+CSV reports

reports/
  eval_<ts>.json         Per-case evaluation payload
  eval_<ts>.csv          Flat per-case summary
```

## Agent pipeline

```
question -> Router -> Schema -> SQL gen -> Safety -> DuckDB
                                                       v
                                       Visualization + Recommendation
                                                       v
                                                Streamlit UI
```

1. **Router agent** is **LLM-first**: when an LLM provider is available
   the model is asked for a JSON-validated `RouterOutput` (intent +
   structured filters) using Ollama's native `format="json"` mode. The
   deterministic keyword extractor still runs as a *fact veto* on
   `max_rent`, `min_bedrooms`, `exact_bedrooms`, and `neighborhoods` so
   small-model number/named-entity hallucinations are corrected. When the
   LLM is unreachable or returns invalid JSON the agent falls back to
   pure keyword routing and the public output shape is unchanged.
2. **Clarification agent** is **LLM-first**: the model receives the
   user's question along with the dimensions already extracted and
   returns a `ClarificationDecision` (specific-enough flag + missing
   dimensions + follow-up text) in a single call. The deterministic
   `count_dimensions` check still runs as a sanity guard - the LLM
   cannot claim the question is specific when zero dimensions were
   extracted. In Mock Demo Mode and on LLM failure the agent uses
   `count_dimensions` + a templated follow-up. See
   [Clarification turn](#clarification-turn).
3. **Schema agent** is **deterministic by design**: the canonical
   `src/schema.py` is the only source of truth for tables and columns,
   so the LLM cannot invent names. Proximity filters automatically
   pull in `amenities`, `transit_subway_stations`, or `transit_bus_stops`
   even when the base intent did not require them.
4. **SQL agent** is LLM-first when a provider is configured but every
   LLM-produced statement passes through a strict pre-validator
   (`_validate_llm_sql`): single SELECT/WITH statement, only known
   tables and columns from `src/schema.py`. It also runs a semantic
   guard (`_validate_sql_semantics`) so required location filters
   (borough/neighborhood) extracted by the router cannot be dropped.
   Anything that fails these checks silently falls back to the
   deterministic templates in `_mock_sql`. Mock Demo Mode skips the
   LLM entirely.
5. **Safety agent** is **deterministic by design** - the security
   firewall. It rejects anything that is not `SELECT`/`WITH`, references
   unknown tables, or contains DDL/DML keywords, and auto-appends
   `LIMIT 100` if no limit was specified. An LLM is never allowed to
   decide whether SQL is safe.
6. **Query tool** runs the safe SQL via DuckDB and returns a pandas
   DataFrame.
7. **Visualization agent** is **deterministic by design** - chart type
   (`bar` / `scatter` / `map` / `table_only`) is decided from the
   DataFrame shape and column types. Round-tripping the LLM here would
   add latency without improving correctness.
8. **Recommendation agent** is LLM-first for prose generation: it asks
   the LLM for a 3-5 sentence renter-friendly summary and uses a
   templated fallback in Mock Demo Mode.

### Clarification turn

The clarification agent (`src/agents/clarification_agent.py`) counts the
distinct query dimensions supplied by the user. The dimensions are:

- `max_rent` (price)
- `bedrooms` (`exact_bedrooms` or `min_bedrooms`)
- `neighborhoods` (location)
- `amenities` (parks, places of worship, grocery, schools, ...)
- `proximity` (see below)
- `safety_preference`
- `transit_preference`
- `sort_preference` (e.g. *"by renter_fit_score"*)
- `ranking_signal` (e.g. *"top"*, *"best"*, *"lots of"*)
- `analysis_intent` (specific analysis intents like
  `housing_supply_analysis` count as one dimension on their own)

If fewer than `MIN_PARAMETERS = 2` dimensions are populated, the agent
returns `needs_clarification=True` along with a short follow-up question.
With a real LLM provider the question is generated on the fly; in mock
mode a deterministic template is used. The Streamlit Ask AI tab renders
the conversation as `st.chat_message` bubbles and feeds the user's reply
back through the router via `combine_turns(prior_turns + [reply])`.

### Proximity filter

The router extracts a list of `{target, max_distance_miles, kind}`
proximity entries from natural-language phrases such as:

- `within 0.5 mi of a mosque` (explicit distance)
- `closer than 1 km of a park` (km is converted to miles)
- `within 3 blocks of a church` (blocks counted as 0.05 mi each)
- `near subway`, `close to a bus stop`, `walking distance to a train`
  (default radius **0.25 mi** for both subway and bus)

Two proximity flavours are emitted:

- `kind="amenity"` -> SQL adds an `EXISTS` subquery over the
  pre-populated `amenities.distance_miles` column.
- `kind="transit"` (`subway` or `bus`) -> SQL adds an apartment-level
  haversine scalar subquery against `transit_subway_stations` /
  `transit_bus_stops` and also returns the distance as a
  `distance_to_subway_miles` / `distance_to_bus_miles` column. This is a
  true point-to-point distance per listing, distinct from the aggregate
  `transit_score`. For neighborhood-level intents without an apartments
  join the agent falls back to the precomputed
  `ns.nearest_subway_distance` / `ns.nearest_bus_stop_distance` columns.

## Methodology

The pipeline mixes LLM-driven reasoning with deterministic Python in a
specific trust hierarchy:

**LLM-first agents** (Ollama drives, deterministic code is the validator
or the fallback):

- **Router** - the LLM produces a Pydantic-validated `RouterOutput`
  (intent + structured filters). Keyword extraction runs in parallel
  and *vetoes* the LLM only on high-stakes facts (`max_rent`,
  `min_bedrooms`, `exact_bedrooms`, `neighborhoods`) where small models
  routinely fudge numbers and named entities. On LLM failure (timeout,
  invalid JSON, wrong shape) the agent falls back to the keyword
  extractor.
- **Clarification** - the LLM judges whether the question is specific
  enough and writes the follow-up text in a single call. The
  deterministic `count_dimensions` check vetoes a "specific enough"
  verdict when zero dimensions were actually extracted.
- **SQL** - the LLM proposes the query; `_validate_llm_sql` enforces
  single SELECT/WITH + known tables + known columns *before* the
  query reaches the safety agent or DuckDB. On any pre-validator
  failure the agent silently falls back to the templated `_mock_sql`.
- **Recommendation** - the LLM writes the renter-friendly summary;
  Mock Demo Mode uses a deterministic template.

**Deterministic-by-design agents** (LLM is never consulted):

- **Schema agent** - the canonical schema in `src/schema.py` is the
  only source of truth, so an LLM cannot invent table or column names.
- **Safety agent** - the SQL firewall. Code-only validation guarantees
  the database only ever sees one read-only `SELECT`/`WITH` statement
  against a known table. Letting an LLM decide whether SQL is safe
  would defeat the security boundary.
- **Visualization agent** - chart selection is a function of DataFrame
  shape and column types; round-tripping the LLM here would add
  latency without improving correctness.

Additional scoring assumptions used in the app:

- **HPD building data is a housing-stock signal**, not direct vacancy.
- **Transit and amenity values are access/proximity signals**, not service guarantees.
- **Missing source datasets default affected scores to `50`** so the app stays usable while surfacing warnings.

This split means a malformed JSON response from a small local model
silently degrades the LLM-first agents to their deterministic fallbacks
without the user noticing - and the deterministic-by-design agents
remain untouched so the security and correctness floors hold even if
the model misbehaves. Mock Demo Mode (`provider == "mock"`) skips every
LLM call so the entire app is reproducible offline.

The Foundation layer powering this is in `src/llm_client.py`
(`generate_validated(model_cls)` + Ollama `format="json"`) and
`src/agents/schemas.py` (Pydantic schemas for `Filters`,
`RouterOutput`, and `ClarificationDecision`).

## Score recipes

```
amenity_score        = 0.30*transit + 0.25*green_space + 0.20*public_service
                       + 0.15*worship + 0.10*grocery
                       (weights redistributed if worship/grocery missing)

availability_score   = 0.60*listings_norm + 0.25*housing_supply
                       + 0.15*affordable_housing_signal
                       (no listings: 0.70*housing_supply + 0.30*affordable_housing_signal)

renter_fit_score     = 0.30*affordability + 0.25*safety + 0.25*amenity
                       + 0.20*availability

value_score          = 0.50*affordability + 0.30*safety + 0.20*amenity

safety_affordability_score = 0.50*affordability + 0.50*safety
```

### Personalized requested-score ranking

For Ask AI apartment results, AI-partments computes request-aware match
scores after SQL returns rows:

- `price_match_score` from `rent` vs requested `max_rent` (with over-budget penalty)
- `safety_match_score` from `safety_score` (or inverse `crime_score` fallback)
- `transit_match_score` from distance-to-transit columns (or `transit_score` fallback)
- `proximity_match_score` from best available requested-distance signal
- `amenity_match_score` from `amenity_score`

An `overall_match_score` is then computed as a weighted average over available
component scores:

- requested dimensions weight = `2.0`
- non-requested available dimensions weight = `1.0`

Apartment results are ordered by `overall_match_score DESC`, and the table
displays only requested component score columns plus `overall_match_score`
(non-requested score columns are hidden).

When a source dataset is missing, the affected score falls back to a neutral
50 and the pipeline emits an explicit warning that surfaces in the UI.

## Smoke test

```bash
python scripts/smoke_test.py
python scripts/test_safety.py
python scripts/test_clarification.py
python scripts/test_router_intent.py
python scripts/evaluate_pipeline.py
```

`smoke_test.py` runs the seven canonical demo questions end-to-end.
`test_safety.py` exercises the SQL safety validator (including the new
amenity `EXISTS` and transit haversine subqueries). `test_clarification.py`
covers `count_dimensions`, the proximity extractor, and the clarification
flow including a multi-turn `combine_turns` follow-up.
`test_router_intent.py` runs the canned dashboard examples through the
router agent in `mock` and `ollama` modes side by side so you can see
what the LLM-first inversion changes vs. the keyword fallback.

`evaluate_pipeline.py` is the deterministic agentic evaluation harness.
For every case it asserts properties on the router output, clarification
result, schema selection, generated SQL substrings, safety verdict,
returned DataFrame (including `distance_to_subway_miles` /
`distance_to_bus_miles` bounds for transit proximity cases), chart type,
and recommendation. Reports are written to:

```
reports/eval_<timestamp>.json   # full per-case payload
reports/eval_<timestamp>.csv    # one summary row per case
```

Run against a real LLM with `python scripts/evaluate_pipeline.py --provider ollama`
or `--provider watsonx`.

## License

MIT (or whatever your hackathon prefers).
