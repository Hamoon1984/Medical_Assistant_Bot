# Medical Assistant Bot

## Overview
This project implements a simple **medical question-answering system** using NLP techniques. The bot uses **TF-IDF vectorization** and **cosine similarity** to retrieve the most relevant answer from a dataset of medical questions and answers.

---

## Dataset
We use the provided dataset `mle_screening_dataset.csv` containing medical Q&A pairs on diseases such as **Glaucoma** and **High Blood Pressure**. Duplicate Q&A pairs were removed before training.

---

## Important note about data and Dataset 
Please copy / paste the data in the data folder before running. I did not put the data in git since it is not a good practice.

---


## Approach
1. **Data Preprocessing**:
   - Removed duplicate question-answer pairs.
   - Split dataset into training (80%) and testing (20%) sets.

2. **Model**:
   - Used **TF-IDF Vectorization** to represent questions numerically.
   - Applied **Cosine Similarity** to find the closest matching question in the training set.
   - Returned the corresponding answer.

3. **Evaluation**:
   - **Mean Reciprocal Rank (MRR)**: Measures ranking quality of the correct answer among retrieved results.

---

## How to run the code?


### Setting up the mle_screening_env Python 3 Virtual Environment using anaconda
 * Create a Virtual Environment using conda - `conda create -n mle_screening_env python=3.10`
 * To activate this environment, use, `conda activate mle_screening_env`
### Cloning the Medical_Assistant_Bot Repo
 * Clone Medical_Assistant_Bot Repo - `git@github.com:Hamoon1984/Medical_Assistant_Bot.git` (Make sure you've added your SSH Keys to Gitlab)

### Install dependencies
 * Install all the packages (This will take a few minutes) - `pip install -r requirements.txt`
```console
$ cd Medical_Assistant_Bot
$ pip install -r requirements.txt
```
* Run the code
```console
$ cd medical_assistant
$ python medical_assistant_bot.py
```
### Setting up the mle_screening_env Python 3 Virtual Environment without anaconda in windows if you have license issue with conda
 * Install python 3.10.0 (https://www.python.org/downloads/release/python-3100/)
```console
$ cd path_to_python_3.10
$ python -m venv mle_screening_env
$ cd mle_screening_env/Scripts
$ activate.bat (in cmd)
```
 * move the requirements.txt to the Scripts folder 
```console
$ pip install -r requirements.txt
```

### Example Usage in cmd or bash (Make sure the virtual environment is activated.)
```
python medical_assistant_bot.py
```
### Future works
 * This chat bot could be served with a Flask App but it is not developed here yet

