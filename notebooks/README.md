# Article notebooks

[Sentiment analysis: Word2Vec to BERT](sentiment-analysis-from-word2vec-to-bert.ipynb) accompanies the [article](https://nanx.cc/2020/04/25/sentiment-analysis-from-word2vec-to-bert.html).

- **[Read the completed notebook on nbviewer](https://nbviewer.org/github/xia0nan/xia0nan.github.io/blob/master/notebooks/sentiment-analysis-from-word2vec-to-bert.ipynb)** — view the code, saved tables, plot, and training output without running anything.
- [Open the shared Colab notebook](https://colab.research.google.com/drive/1tFnKYPX37N7kmoqgKaEtL0qfo1E1DK33?usp=sharing) — optional, for further experiments.

The Colab link points to the shared run. Choose **File → Save a copy in Drive** before experimenting. The nbviewer link displays the repository snapshot and works once the notebook is published to `master`.

The saved Colab run completed on 22 September 2026 (Singapore), using `MODE = "quick"`, 2,000 training reviews, 500 validation reviews, seed 42, and one neural epoch. `RUN_TEST = False`; there are no test-set results. The runtime output reports Python 3.13.15, Transformers 5.16.1, and PyTorch 2.11.0+cu128; notebook metadata identifies a T4 GPU.

Code cells, execution counts, outputs, and warnings are preserved from the supplied Colab export. Only explanatory Markdown was updated; the notebook was not rerun during integration. The BERT output contains checkpoint-loading warnings about LayerNorm parameter names, so its recorded score is provisional. See the notebook's opening summary and the article for the results and limitations.

To experiment in Colab, select a GPU runtime, run setup, restart the session, and continue from configuration. The executed setup uses unpinned packages, so a future environment may differ. Keep test evaluation disabled while choosing settings. Save your executed copy and download its run directory before ending the session.

The separate run manifest, split IDs, prediction CSVs, and model checkpoints are not included in this repository. Keep saved notebook outputs when editing this walkthrough; do not commit generated caches, model checkpoints, or datasets to the website repository.
