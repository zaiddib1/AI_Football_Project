# Unsupervised player styles

Open **`player_style_clustering.ipynb`**, a separate notebook from the position classifier.
It clusters the 180-minute outfield cohort using ten standardized event rates. Positions,
names, clubs, leagues, age, and exposure are excluded from the clustering inputs. Position
labels return only for the comparison after cluster assignments and event-based names are fixed.

The notebook includes cluster profiles, alternative cluster counts, resampling stability,
feature/exposure sensitivity, representative players, and a comparison to official positions.
Five clusters provide exploratory granularity; the separation checks may favor fewer broad
groups. Clusters are fitted in standardized rate space, and UMAP is a display of those records.

From the enclosing FootballAIProject directory:

```sh
.venv/bin/python -m pip install -r AI_Football_Project/requirements-clustering.txt
.venv/bin/python scripts/execute_style_notebook.py
```

Select the same Python environment in Jupyter/VS Code and Run All. The first UMAP fit may
take longer because of numerical compilation. Rebuilding notebook source is optional:
`scripts/build_player_style_notebook.py` writes the notebook without outputs; execute it afterward.
No existing classifier notebook, cleaned CSV, or extracted source file is replaced.

The notebook creates `style_clustering_results/player_styles_umap.html`. Open this file in a
browser if the notebook viewer blocks embedded JavaScript. It works offline and includes
player search, league/minutes filters, style/official-position colors, three neighborhood
layouts, hover statistics, and zoom. Neither filters nor color choices refit the model.

Each point is a player–club record. A handful of repeated names are kept separately because
the data lacks persistent player IDs and match dates; the hover includes club information.
The source statistics/club membership are not independently verified. Rates can be noisy
at low exposure, and apparent UMAP islands should not be treated as proof of natural groups.

Full CSV tables, player assignments and coordinates, PNG figures, the exact standardized
feature matrix, and `clustering_manifest.json` are saved beside the map. Identifiers and
official positions in the assignment export are metadata for exploration, not model inputs.
