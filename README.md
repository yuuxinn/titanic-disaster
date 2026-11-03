# Titanic Disaster – Survival Prediction (Python & R)

This repo predicts which passengers survived the Titanic disaster (April 15, 1912) using **logistic regression**, implemented twice:

- **Python** (pandas + scikit-learn), in `src/python/`
- **R** (dplyr + base `glm`), in `src/r/`

Each version runs inside its own **Docker container**. It loads the data, explores it, cleans it, engineers features, trains the model, reports accuracy, and predicts on the test set. Every step is printed to the terminal.

> **The dataset is not included in this repo.** Follow Step 2 below to download it.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS            # repo owner must review pull requests
├── .gitignore                # ignores the data folder, venvs, OS files
├── README.md
└── src/
    ├── data/                 # put train.csv and test.csv here (git-ignored)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt  # pinned package versions
    │   └── main.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

---

## Prerequisites

| Tool | Why | Get it |
|------|-----|--------|
| Git | to clone the repo | https://git-scm.com/downloads |
| Docker Desktop | to build and run the containers | https://www.docker.com/products/docker-desktop/ |
| Kaggle account | to download the data | https://www.kaggle.com |

Make sure Docker Desktop is **open and running** before Step 3. Check with:

```bash
docker --version
```

You do **not** need Python or R installed on your machine. Docker handles everything.

---

## Step 1 – Clone the repo

```bash
git clone https://github.com/yuuxinn/titanic-disaster.git
cd titanic-disaster
```

**Run every command below from this `titanic-disaster` folder (the repo root).**

---

## Step 2 – Download the data

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. If prompted, click **Join Competition** / **I Understand and Accept**. Kaggle requires this before it allows downloads.
3. Click **Download All**. This gives you `titanic.zip`.
4. Unzip it and move **`train.csv`** and **`test.csv`** into the `src/data/` folder of this repo.

*Alternative (Kaggle CLI):* if you have the Kaggle API set up (`pip install kaggle` plus a `kaggle.json` token), run:

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

Your folder should now look like this:

```
src/data/
├── .gitkeep
├── gender_submission.csv   (optional, not used)
├── test.csv
└── train.csv
```

---

## Step 3 – Run the Python version

**Build the image** (takes 1–2 minutes the first time):

```bash
docker build -t titanic-python src/python
```

**Run the container.** The data folder is mounted into the container, so nothing gets copied into the image.

macOS / Linux / Git Bash:
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

Windows PowerShell:
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

Windows Command Prompt:
```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

---

## Step 4 – Run the R version

**Build the image** (takes a few minutes the first time while R packages install):

```bash
docker build -t titanic-r src/r
```

**Run the container:**

macOS / Linux / Git Bash:
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

Windows PowerShell:
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

Windows Command Prompt:
```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-r
```

---

## What you will see

Both scripts print the same six steps:

1. **Load training data** – shape and first 5 rows of `train.csv`
2. **Explore** – missing values per column and survival rates by sex and class
3. **Clean and engineer features**, with a printed line for each change:
   - `Title` extracted from `Name` (Mr, Mrs, Miss, Master, Rare)
   - missing `Age` filled with the median age for that title
   - missing `Fare` / `Embarked` filled with the training median / mode
   - new features `HasCabin`, `FamilySize`, `IsAlone`
   - categorical columns encoded
4. **Train the logistic regression model** – hold-out validation accuracy (80/20 split of `train.csv`), training accuracy, and model coefficients
5. **Predict on `test.csv`** – predictions are saved to `src/data/predictions_python.csv` or `src/data/predictions_r.csv`
6. **Test accuracy** – see the note below

### Note on test-set accuracy

Kaggle's `test.csv` **does not include the `Survived` column**. The true labels are kept private for the competition, so accuracy on the test set cannot be computed locally. To deal with this, the scripts:

- report a **hold-out validation accuracy** (the model is trained on 80% of `train.csv` and scored on the other 20%) as an honest estimate of performance on unseen data;
- save the test predictions in Kaggle's submission format. Upload `predictions_python.csv` or `predictions_r.csv` at https://www.kaggle.com/competitions/titanic/submissions to get the official test accuracy.

If a `test.csv` that includes `Survived` is supplied, the scripts automatically print the real test accuracy instead.

### Model differences

- **Python** uses all features, including `Sex`. scikit-learn's logistic regression is regularized by default, so overlapping features are handled fine.
- **R** uses base `glm`, which is not regularized. `Sex` is left out because `Title` already encodes it (Mr vs Mrs/Miss). Keeping both makes the coefficient estimates unstable.

---

## Optional – run Python locally without Docker

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r src/python/requirements.txt
python src/python/main.py
```

The `venv/` folder is git-ignored.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: could not find train.csv` | The data isn't in `src/data/`, or you ran `docker run` from a folder other than the repo root. `cd` into `titanic-disaster` and check that `src/data/train.csv` exists. |
| `Cannot connect to the Docker daemon` | Docker Desktop isn't running. Open it and wait until it says it's running. |
| Mount path error on Windows | Use the PowerShell or Command Prompt version of the command above, matching the terminal you are in. |
| R build is very slow or fails on an Apple Silicon Mac | Build and run for Intel instead: `docker build --platform linux/amd64 -t titanic-r src/r`, then add `--platform linux/amd64` to the `docker run` command. |
| Want to rebuild from scratch | `docker build --no-cache -t titanic-python src/python` (same for `titanic-r`). |
