<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E86AB,100:A23B72&height=220&section=header&text=ChatGPT%20Data%20Analytics&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Decoding%2052%2C000%2B%20real%20prompts%20%E2%80%94%20what%20do%20people%20actually%20ask%20ChatGPT%3F&descAlignY=60&descSize=16&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Dataset-Alpaca--GPT4-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)

<br/>

> **52,002 real ChatGPT instruction–response pairs. Six analytical lenses. One complete picture of how humans talk to AI.**

<br/>

</div>

---

## 📖 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🗂️ Dataset](#️-dataset)
- [🔬 Analysis Sections](#-analysis-sections)
  - [☁️ Word Cloud — What Words Dominate?](#️-word-cloud--what-words-dominate)
  - [🏷️ Prompt Classification — What Do People Ask For?](#️-prompt-classification--what-do-people-ask-for)
  - [📖 Readability — How Clear Are ChatGPT's Answers?](#-readability--how-clear-are-chatgpts-answers)
  - [📏 Prompt Length vs Readability](#-prompt-length-vs-readability)
  - [🗣️ Verbosity — How Wordy Is ChatGPT?](#️-verbosity--how-wordy-is-chatgpt)
  - [📎 Context Effect — Does Extra Info Help?](#-context-effect--does-extra-info-help)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [💡 Key Findings](#-key-findings)
- [🙋 Author](#-author)

---

## 📌 Project Overview

This project is a **deep exploratory data analysis** of real-world ChatGPT usage patterns, powered by the publicly available **Alpaca-GPT4** dataset on HuggingFace.

The core question this notebook answers:

> *When people sit down to type a prompt into ChatGPT — what do they write, how do they write it, and what kind of answer do they get back?*

Rather than studying ChatGPT's capabilities in isolation, this project studies the **human side of the interaction** — the language patterns, task preferences, and structural habits of 52,002 real prompts — then links those patterns to properties of ChatGPT's responses.

**This is not a model evaluation. It is a behavioural analysis.**

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Source** | [`vicgalle/alpaca-gpt4`](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) on HuggingFace |
| **Format** | Parquet (loaded directly via URL) |
| **Records** | 52,002 instruction–response pairs |
| **Columns** | `instruction` (the prompt), `input` (optional extra context), `output` (ChatGPT's response) |
| **Origin** | GPT-4 generated responses to Stanford Alpaca-style prompts |

```python
df = pd.read_parquet("hf://datasets/vicgalle/alpaca-gpt4/data/train-00000-of-00001-6ef3991c06080e14.parquet")
df.shape  # → (52002, 3)
```

---

## 🔬 Analysis Sections

### ☁️ Word Cloud — What Words Dominate?

**Goal:** Visualise the most frequent meaningful words across all 52,002 prompts after removing noise words.

**Approach:**
- Combined NLTK's built-in English stop words with a custom list of instruction-verb noise words (`write`, `generate`, `create`, `give`, `list`, etc.)
- Applied lowercase normalisation and regex cleaning to strip punctuation
- Generated a word cloud from the merged, filtered text corpus

<br>
<img width="640" height="329" alt="wordcloud" src="https://github.com/user-attachments/assets/12bf130e-72fb-4568-bb00-bde95d10634a" />
<br>

**What image shows:**

| Word Size | Meaning |
|-----------|---------|
| 🔵 Giant — *design, story, item* | These themes dominate across thousands of prompts |
| 🟢 Large — *question, data, character, calculate* | Very common task-type signals |
| 🟡 Medium — *poem, technology, analyze, convert* | Moderately frequent topics |

> **Insight:** People primarily use ChatGPT for **creative content** (story, design, character) and **informational tasks** (question, difference, describe). Technical tasks like mathematics appear far less frequently than intuition might suggest.

---

### 🏷️ Prompt Classification — What Do People Ask For?

**Goal:** Categorise every prompt into a task type using rule-based keyword classification.

**Method:** A custom `classify_prompt()` function checks the start of each instruction against keyword clusters:

```
Question       → "can you", "could you", "should i", "is it"...
Creative Task  → "write", "create", "generate", "compose", "draft"
Explanation    → "explain", "describe", "define", "clarify"
Listing Task   → "give", "list", "provide", "name", "outline"
Advice         → "suggest", "recommend", "tips for", "ways to"
Editing        → "rewrite", "rephrase", "improve", "fix"
Classification → "classify", "categorize", "group the following"
Problem Solving→ "calculate", "solve", "compute", "evaluate"
Other          → everything else
```
<img width="1038" height="500" alt="chat_distributio" src="https://github.com/user-attachments/assets/4fe22180-b8bd-4a1f-af6c-de0594f7e2b3" />
<br>

**Results:**

| Rank | Prompt Type | Count | Share |
|---|---|---|---|
| 🥇 | Creative Task | 11,877 | ~22.8% |
| 🥈 | Listing Task | 7,328 | ~14.1% |
| 🥉 | Explanation | 5,066 | ~9.7% |
| 4 | Editing/Rewriting | 1,980 | ~3.8% |
| 5 | Classification | 1,389 | ~2.7% |
| 6 | Advice | 1,122 | ~2.2% |
| 7 | Problem Solving | 784 | ~1.5% |
| 8 | Question | 117 | ~0.2% |

> **Insight:** ChatGPT is overwhelmingly used as a **creative and organisational tool**, not a question-answering engine. Direct questions — the form most people associate with "AI assistants" — are actually the rarest prompt type.

---

### 📖 Readability — How Clear Are ChatGPT's Answers?

**Goal:** Measure the reading complexity of every ChatGPT response using the **Flesch Reading Ease** scale.

**The Flesch Reading Ease score** rates text from 0 to 100+ based on sentence length and syllable count:

| Score | Reading Level |
|---|---|
| 90–100 | Very Easy — 5th grade |
| 70–89 | Easy — conversational |
| 50–69 | **Medium — standard/newspaper** ✅ Most ChatGPT responses |
| 30–49 | Difficult — academic |
| 0–29 | Very Difficult — technical/legal |
| < 0 | Extremely Difficult |
<br>
<img width="1276" height="450" alt="readability_level" src="https://github.com/user-attachments/assets/ff5ed39a-57dd-4d25-9565-348faccae8e8" />
<br>

**What the pie chart shows:**

- **Medium (44.5%)** — Almost half of all responses sit at a clear, professional reading level
- **Easy (28.6%)** — Over a quarter are genuinely accessible to a general audience
- **Difficult (15.3%)** — About 1 in 6 responses tips into complex territory
- Extreme categories (Very Easy, Very Difficult, Extremely Difficult) are rare edge cases

> **Insight:** ChatGPT defaults to a **moderate, accessible writing style** for most tasks. It neither over-simplifies nor over-complicates — the distribution closely resembles well-edited magazine or blog writing.

---

### 📏 Prompt Length vs Readability

**Goal:** Test whether longer prompts produce simpler or more complex responses.

**Method:**
1. Count the word count of every instruction (`instruction_word_count`)
2. Compute the Pearson correlation between word count and Flesch score
3. Fit a linear regression line and extract slope (m) and intercept (c)
4. Visualise with both a seaborn regression plot and an interactive Plotly scatter
<br>

<img width="1276" height="450" alt="length_vs_readability_score" src="https://github.com/user-attachments/assets/403ef045-96ed-4b6f-b584-ffa379f7e7ef" />
<br>

**What the charts show:**

- The **regression line is nearly flat** — barely rising from left to right
- Data points are **scattered uniformly** across all prompt lengths with no clustering pattern
- The **correlation coefficient is close to zero** (roughly +0.05 to +0.15)

> **Insight:** There is **no meaningful relationship** between prompt length and response readability. Whether you write a 5-word or 50-word prompt, ChatGPT calibrates response complexity based on the *topic and intent* of your request — not how many words you used to ask it.

---

### 🗣️ Verbosity — How Wordy Is ChatGPT?

**Goal:** Measure how many words ChatGPT uses per sentence across all responses.

**Method:**
- Sentence count extracted using regex splitting on `.`, `!`, `?` patterns
- Word count computed using `.split()`
- Words per sentence = `output_word_count / sentence_count` (with division-by-zero guard)
- Visualised as a Plotly box plot with overlaid data point strip

<img width="770" height="613" alt="instruction_length vs flash_score" src="https://github.com/user-attachments/assets/ddc8a13c-3be2-4f66-8e52-ce2c5c39017a" />
<br>


**What the box plot shows:**

| Statistic | Value |
|---|---|
| Median words/sentence | ~15–20 words |
| Typical range (IQR) | ~5 to ~40 words/sentence |
| Maximum outliers | 370+ words/sentence |

The **box is narrow and sits low** — most responses use concise, readable sentences. However, a significant tail of extreme outliers exists, driven by responses containing bullet-pointed lists, numbered steps, or code blocks, which the sentence splitter may count as a single giant sentence.

> **Insight:** ChatGPT typically writes in **clear, moderately-sized sentences** consistent with professional writing standards. The extreme outliers are artefacts of structured formatting, not genuinely run-on prose.

---

### 📎 Context Effect — Does Extra Info Help?

**Goal:** Compare ChatGPT responses for prompts that included extra context (`input` column) versus those that did not.

**Method:**
- Created binary flag `has_input` (1 = context provided, 0 = no context)
- Grouped by this flag and computed mean `output_word_count` and mean `flesch_score`
- Visualised as a grouped bar chart
<br>
<img width="950" height="600" alt="extra_text vs prompt_response" src="https://github.com/user-attachments/assets/3618bb8e-7d28-4ae8-9244-d0692a18d792" />
<br>

**Results:**

| Group | Avg. Word Count | Avg. Flesch Score |
|---|---|---|
| No extra context | **138.48 words** | 46.36 |
| Extra context provided | **66.07 words** | **53.21** |

> **Insight:** This is the most surprising finding in the project. Providing extra context produces responses that are **52% shorter** and **noticeably more readable**. Specificity helps ChatGPT stay focused. Vague prompts generate long, generic answers; precise prompts generate shorter, cleaner ones — **quality over quantity**.

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│  Data Layer          │  pandas, numpy, pyarrow           │
│  NLP & Text          │  nltk, re, textstat, wordcloud    │
│  Static Plots        │  matplotlib, seaborn              │
│  Interactive Plots   │  plotly express                   │
│  Environment         │  Jupyter Notebook / JupyterLab    │
│  Dataset Source      │  HuggingFace Hub (Parquet)        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/doyancha/CHAT-GPT-DATA-ANALYTICS.git
cd CHAT-GPT-DATA-ANALYTICS
```

### 2. Install dependencies

```bash
pip install pandas numpy nltk wordcloud textstat matplotlib seaborn plotly pyarrow
```

Or install in Jupyter directly:

```python
# These are included as commented cells at the top of the notebook
# !pip install nltk wordcloud textstat plotly pyarrow
```

### 3. Download NLTK data

```python
import nltk
nltk.download("stopwords")
```

### 4. Run the notebook

```bash
jupyter notebook CHAT_GPT_DATA_ANALYTICS_ANNOTATED.ipynb
```

> ⚠️ **Note:** The dataset is loaded directly from HuggingFace via a public Parquet URL. An active internet connection is required for the data loading cell.

---

## 📁 Project Structure

```
CHAT-GPT-DATA-ANALYTICS/
│
├── 📓 CHAT_GPT_DATA_ANALYTICS_ANNOTATED.ipynb   ← Main analysis notebook
├── 📄 README.md                                  ← You are here
└── 📄 LICENSE                                    ← MIT License
```

---

## 💡 Key Findings

Here are the six headline insights from this analysis, in plain English:

```
🔤  1. WHAT PEOPLE WRITE ABOUT
     "Design", "story", and "item" dominate ChatGPT prompts.
     Creative content requests far outnumber technical or factual ones.

🏷️  2. MOST POPULAR TASK TYPE
     Creative Tasks (11,877 prompts) are nearly 2× more common than
     Listing Tasks (7,328), the second-place category.
     Direct questions are the rarest task type of all.

📖  3. HOW READABLE ARE THE ANSWERS
     44.5% of responses sit at a "Medium" reading level — clear,
     professional, and accessible to most adults.

📏  4. DOES PROMPT LENGTH MATTER FOR CLARITY?
     No. Correlation ≈ 0. Writing more words in your prompt does
     not make ChatGPT's response easier or harder to read.

🗣️  5. HOW VERBOSE IS CHATGPT?
     Median ~15–20 words per sentence. Concise and professional.
     Extreme outliers are caused by lists and code, not rambling prose.

📎  6. THE POWER OF CONTEXT
     Prompts with extra context get responses that are 52% shorter
     and more readable. Specificity = better answers.
```

---

## 🙋 Author

<div align="center">

**Built and analysed by [@doyancha](https://github.com/doyancha)**

*If you found this project useful or interesting, please consider giving it a ⭐ — it helps others discover the work.*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-doyancha-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/doyancha)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:A23B72,100:2E86AB&height=120&section=footer" width="100%"/>

*Made with 🐍 Python · 📊 Data · 🤖 Curiosity*

</div>
