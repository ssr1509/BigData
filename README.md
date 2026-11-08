# AI-Generated Text Detection: Robustness Experiments

This project studies how well AI-generated-text detectors work when text is changed by different attacks and when the detectors are tested on a separate dataset.

The experiments use:

- **RAID** for training, validation, and the main attack-based evaluation.
- **MAGE** for out-of-dataset evaluation, including a GPT-4 test set and a GPT-4 paraphrase test set.
- **GPT-2** as a zero-shot likelihood baseline.
- **RoBERTa-base** for three supervised classifiers: Naive, Robust (partial), and Robust-all.

The main experiment is implemented in `raid_detector_robustness_final.ipynb`.

---

## 1. Repository contents

Keep the following files in the project directory:

```text
project/
├── raid_detector_robustness_final.ipynb
├── README.md
└── data/                         # Created during execution, if not already present
    ├── raid_sample.*             # Sampled RAID data
    ├── raid_cleaned.*            # Cleaned and split RAID data
    └── mage/                     # Downloaded MAGE test files, if saved locally
```

The exact intermediate filenames and output locations are determined by the notebook's configuration cells. If your copy of the notebook uses different names, use the paths specified there.

**Important:** The notebook downloads and processes a large RAID source file. Do not assume that the intermediate data are included in the repository. They may need to be generated again.

---

## 2. Software and libraries

The workflow uses Python and the following main packages:

| Package / tool | Purpose |
|---|---|
| Python 3 | Runs the notebook and experiment code |
| Jupyter Notebook or JupyterLab | Executes the notebook |
| PyTorch | Runs GPT-2 and RoBERTa models |
| Hugging Face Transformers | Loads GPT-2, RoBERTa, and their tokenizers |
| PySpark | Processes, cleans, and splits the sampled RAID data |
| Pandas | Handles smaller data tables and result summaries |
| PyArrow | Reads and writes Parquet data, where used |
| scikit-learn | Calculates evaluation metrics such as AUROC |
| NumPy | Numerical operations |
| tqdm | Progress bars, where used |

The notebook may also import standard Python packages such as `os`, `random`, `math`, `json`, and `collections`.

### Install the main dependencies

Create and activate a virtual environment if you are running locally:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the main packages:

```bash
python -m pip install --upgrade pip
pip install jupyterlab ipykernel
pip install torch transformers
pip install pyspark pandas pyarrow
pip install scikit-learn numpy tqdm
```

If the notebook reports a missing package, install that package in the same environment and restart the notebook kernel.

> **Version note:** The notebook's exact package versions were not recorded in this README. For strict reproducibility, record the versions from the environment used for the final run with `pip freeze > requirements-lock.txt`. Reuse that file in later runs with `pip install -r requirements-lock.txt`.

### Hardware

- A CUDA-compatible NVIDIA GPU is recommended for model scoring and RoBERTa training.
- The notebook can use CPU execution where supported, but model inference and training will be substantially slower.
- The RAID source data and intermediate files require considerable disk space. Make sure there is enough free storage before starting.
- PySpark also needs enough memory for the data-processing stage. Avoid assigning all system memory to Spark; leave memory available for Python, the operating system, and model execution.

---

## 3. Data sources

### 3.1 RAID

RAID is the main dataset used for training, validation, and attack-based testing.

The notebook downloads or reads the RAID training CSV (`train.csv`). The source file is large (approximately **11.8 GB** in the run documented during this project). The notebook samples from the source rather than training directly on the complete CSV.

The experiment includes the clean condition (`none`) and 11 attack conditions:

- `alternative_spelling`
- `article_deletion`
- `homoglyph`
- `insert_paragraphs`
- `number`
- `paraphrase`
- `perplexity_misspelling`
- `synonym`
- `upper_lower`
- `whitespace`
- `zero_width_space`

The clean `none` condition is not an attack; it is the unmodified reference condition.

The RAID data also contain domain and source information. The notebook uses fields such as the model/generator, domain, attack, source identifier, and generated text during preparation and evaluation.

### 3.2 MAGE

MAGE is used only for external evaluation. Its examples are **not** used to train the RAID models or to calibrate the model thresholds.

The notebook evaluates two MAGE files:

1. `test_ood_set_gpt.csv` — the GPT-4 out-of-dataset test set.
2. `test_ood_set_gpt_para.csv` — the GPT-4 paraphrase out-of-dataset test set.

The notebook reads the text, label, and source (`src`) fields where available. It converts the MAGE labels to the same convention used for RAID:

- `0` = human-written
- `1` = AI-generated

The MAGE source convention is reversed, so the notebook applies `1 - label` before calculating metrics.

---

## 4. Big-data processing: why Spark is used

The RAID source CSV is too large to treat as a small, ordinary notebook table. The workflow therefore reduces the data before model training.

The data-preparation workflow is:

1. Read the large RAID CSV in chunks.
2. Randomly sample rows while reading the source data. The documented sampling probabilities are 0.5 for human rows and 0.012 for AI-generated rows.
3. Save the sampled data so that later steps do not need to scan the full source CSV again.
4. Use **PySpark** for cleaning and preparing the sampled data.
5. Create a binary label:
   - Human-written text: `0`
   - AI-generated text: `1`
6. Use the generated text as the model input.
7. Remove rows with text shorter than 30 words.
8. Remove duplicate identifiers and duplicate text.
9. Balance the two classes by down-sampling the larger class.
10. Split the data by `source_id`, rather than randomly splitting individual rows. This helps keep related examples from the same source group in one split.
11. Save the processed splits in a reusable format, such as Parquet, as configured in the notebook.

### Recorded data sizes

The following are the counts recorded for the experiment:

| Stage / split | Number of rows |
|---|---:|
| Sampled RAID data before cleaning | 145,677 |
| Cleaned and balanced RAID data | 129,725 |
| Training split | 90,512 |
| Validation split | 19,571 |
| Test split | 19,642 |

The split counts are:

| Split | Human (`0`) | AI (`1`) | Total |
|---|---:|---:|---:|
| Train | 45,302 | 45,210 | 90,512 |
| Validation | 9,815 | 9,756 | 19,571 |
| Test | 9,817 | 9,825 | 19,642 |

The split is approximately 70% training, 15% validation, and 15% testing, based on source groups.

**Why save intermediate files?** Sampling and cleaning the original RAID CSV are expensive steps. Saving the sampled and processed data means that model training can be repeated without downloading and scanning the large source CSV every time.

---

## 5. Model configurations

All three RoBERTa classifiers use the pretrained `roberta-base` checkpoint and a two-class classification head.

| Model | Training examples | Attack exposure |
|---|---:|---|
| GPT-2 zero-shot | Not trained as a RAID classifier | Uses GPT-2 text likelihood as a score |
| Naive RoBERTa | 6,928 | Clean `none` examples only |
| Robust RoBERTa (partial) | 6,928 | Clean text plus homoglyph, zero-width-space, and paragraph-insertion examples |
| Robust-all RoBERTa | 20,784 | Random sample from all RAID conditions, including paraphrase |

The three RoBERTa classifiers use the following recorded settings:

- Base model: `roberta-base`
- Maximum token length: 256
- Epochs: 1
- Batch size: 16
- Learning rate: `2e-5`
- Random seed: `42`

The Robust (partial) model deliberately does **not** train on paraphrase examples. This makes paraphrasing a held-out attack for that model.

The Robust-all model uses three times the training-row budget of the Naive and Robust (partial) models. Its sample is drawn from all attack conditions, but the notebook does not enforce an equal number of examples from every attack. Therefore, its comparison with the other two RoBERTa models changes both attack coverage and total training size.

### GPT-2 baseline

GPT-2 is used as a zero-shot likelihood baseline. The notebook calculates token likelihood scores for each text passage. It does not fine-tune GPT-2 on the RAID labels. The recorded scoring configuration uses a maximum sequence length of 256 tokens and a batch size of 16.

---

## 6. How to run the experiment

### Step 1 — Get the notebook

Place `raid_detector_robustness_final.ipynb` in your working directory.

### Step 2 — Set up the environment

Install the packages listed in Section 2. If you are using a hosted notebook environment, select a runtime with a GPU if one is available.

### Step 3 — Check paths and storage

Open the notebook and inspect its configuration/path cells before running it.

- Confirm where the RAID CSV will be downloaded or read from.
- Confirm the output directory for sampled and cleaned data.
- Confirm the location used for model checkpoints and result files.
- Make sure the storage location has enough free space for the large RAID source file and intermediate files.

Do not change the label mapping or attack names unless you also update the related training and evaluation code.

### Step 4 — Run RAID sampling and preparation

Run the notebook cells in order through the RAID data preparation section.

This stage reads the large source CSV, samples it, cleans the sampled data, balances the labels, creates source-based train/validation/test splits, and saves the intermediate data.

If the notebook has already created the sampled and cleaned files, check the relevant path settings. Reuse those files only if they were generated with the same preprocessing settings and label convention.

### Step 5 — Score GPT-2 and set its threshold

Run the GPT-2 scoring section. The notebook scores the RAID examples and uses clean human examples from the RAID validation split to set the threshold.

### Step 6 — Train the RoBERTa classifiers

Run the training cells for:

1. Naive RoBERTa
2. Robust RoBERTa (partial)
3. Robust-all RoBERTa

Keep the same seed and model settings if you want to reproduce the recorded comparison. Save the trained checkpoints if you want to avoid training them again.

### Step 7 — Evaluate on RAID

Run the RAID evaluation cells. The notebook reports performance by attack type and also includes generator/domain breakdowns where those fields are available.

### Step 8 — Evaluate on both MAGE files

Run the **final combined MAGE evaluation cell**. It evaluates all four detectors on:

- `test_ood_set_gpt.csv`
- `test_ood_set_gpt_para.csv`

It reports AUROC, TPR at the fixed threshold, and FPR at the fixed threshold. It also uses the `src` field for source-wise results where possible.

**Do not run the earlier single-file MAGE evaluation cell as part of the final run.** It evaluates only the paraphrase file and duplicates work performed by the combined cell. Use the combined cell so both MAGE files are evaluated consistently.

### Step 9 — Save the results

Save or export the printed result tables and trained model checkpoints if they are needed for later analysis. Keep the notebook outputs with the run, or save the metrics to CSV files if you want a separate results folder.

---

## 7. Evaluation and threshold setting

Each model receives a score for every text. The project reports:

- **AUROC:** Measures how well the model ranks AI examples above human examples over all possible thresholds.
- **TPR at threshold:** The proportion of AI examples flagged as AI at the selected threshold.
- **FPR at threshold:** The proportion of human examples incorrectly flagged as AI at the selected threshold.

Each model's threshold is selected using the **95th percentile of its scores on clean human examples in the RAID validation split**. This targets approximately 5% false positive rate on that validation group.

The threshold is then kept fixed for RAID test and MAGE evaluation. In particular, **the threshold is not recalibrated on MAGE**. This is intentional: it allows the experiment to measure how a detector calibrated on RAID behaves on a separate dataset.

Because the MAGE data differ from the RAID validation data, the MAGE false positive rate may be higher than 5%. That change is part of the evaluation and should not be hidden by tuning the threshold on MAGE.

---

## 8. Recorded MAGE test-set sizes

The final notebook run recorded the following MAGE counts:

| MAGE file | AI examples | Human examples | Total |
|---|---:|---:|---:|
| GPT-4 OOD | 800 | 762 | 1,562 |
| GPT-4 paraphrase OOD | 1,600 | 762 | 2,362 |

The two files are evaluated separately. Do not merge them into one test set when reporting the main comparison.

---

## 9. Reproducibility notes

For a result that can be checked and repeated:

1. Use the same notebook version and data sources.
2. Keep the random seed at `42`.
3. Keep the recorded sampling probabilities and preprocessing steps.
4. Keep the source-based split rather than making a new row-level random split.
5. Use the same model checkpoints and training settings.
6. Do not change the RAID validation threshold after looking at RAID test or MAGE results.
7. Evaluate both MAGE files separately using the same fixed model thresholds.
8. Record Python and package versions, hardware, runtime type, and date of execution.
9. Preserve the sampled data, cleaned splits, model checkpoints, and result tables where storage allows.

GPU operations and random sampling can produce small differences across hardware, library versions, and execution environments. Exact numerical reproduction may therefore require the same software versions, hardware setup, input files, and saved random samples/checkpoints.

---

## 10. Important interpretation points

- The Naive vs Robust (partial) comparison is the main controlled comparison: both use 6,928 training rows and the same RoBERTa setup. The main difference is whether selected attack examples are included.
- Paraphrase is held out from Robust (partial) training. Its MAGE paraphrase result tests whether training on other selected attacks transfers to paraphrased text from MAGE.
- Robust-all has 20,784 training rows and includes paraphrase examples. Its paraphrase result is **not** an unseen-paraphrase result.
- Robust-all also has a larger training budget than the other RoBERTa classifiers, so its improvement cannot be attributed only to wider attack coverage.
- AUROC and threshold-based TPR/FPR describe different aspects of performance. Report all three rather than relying on only one metric.
- A detector score is not proof of authorship. The results describe performance on the tested datasets and conditions only.

---

## 11. References and dataset papers

- Dugan, L. et al. (2024). **RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.674
- Li, Y. et al. (2024). **MAGE: Machine-generated Text Detection in the Wild.** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.3
- Liu, Y. et al. (2019). **RoBERTa: A Robustly Optimized BERT Pretraining Approach.** https://arxiv.org/abs/1907.11692
- Radford, A. et al. (2019). **Language Models are Unsupervised Multitask Learners.**

---

## 12. Troubleshooting

### The notebook runs out of memory during RAID preparation

- Confirm that the notebook is sampling the source file rather than loading the entire CSV into a Pandas DataFrame.
- Use a machine/runtime with sufficient RAM and disk space.
- Reuse the saved sampled and cleaned Parquet files instead of repeating the full preparation stage.
- If Spark is used locally, check its memory configuration and leave memory for the rest of the Python process.

### Model training or scoring is very slow

- Use a CUDA-enabled GPU runtime if available.
- Confirm that PyTorch detects CUDA.
- Reduce other running processes competing for GPU memory.
- Do not change batch size or sequence length if you are trying to reproduce the recorded run; such changes can affect results.

### CUDA out-of-memory error

- Restart the runtime to release unused GPU memory.
- Check whether another process is using the GPU.
- A smaller batch size may allow execution to continue, but record this change because it differs from the documented batch size of 16.

### Spark cannot start

- Confirm that Java is installed and available to PySpark.
- Restart the notebook kernel after installing or changing PySpark/Java.
- Check that the runtime has enough memory and that the Spark temporary directory has free disk space.

### MAGE AUROC is unavailable for a source group

AUROC requires both AI and human examples. Some `src` groups contain only one class, so AUROC is undefined within those groups. Use the applicable TPR for AI-only groups or FPR for human-only groups; use the overall MAGE table for the two-class AUROC.

---

## 13. Suggested run checklist

- [ ] Correct notebook and data paths are set.
- [ ] Python dependencies are installed.
- [ ] Java/PySpark is available.
- [ ] Enough RAM, disk space, and (ideally) GPU memory are available.
- [ ] RAID sample and processed splits are generated or correctly reused.
- [ ] All three RoBERTa models are trained or the intended checkpoints are loaded.
- [ ] GPT-2 scores and RAID validation thresholds are generated.
- [ ] RAID test evaluation is completed.
- [ ] Both MAGE files are evaluated separately.
- [ ] The final combined MAGE cell is used instead of the older duplicate single-file cell.
- [ ] Result tables, model checkpoints, and environment versions are saved.
