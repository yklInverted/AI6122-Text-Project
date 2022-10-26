# AI6122-Product-Review-Data-Analysis-and-Processing

Group project assignment for AI6122

Submission:

1. Li Kaiyu
2. Chen Lei
3. Li Jiayi
4. Chen Yueqi
5. Chang Lo-Wei

## 1. Prerequisites

The following softwares need to be installed on your system:

- Anaconda: Can be download from https://docs.anaconda.com/anaconda/install/index.html
- Jupyter NoteBook: Can be download via pip using the following command.
```
pip install jupyter notebook
```
- Amazon Product Review Dataset: The datasets are available at https://jmcauley.ucsd.edu/data/amazon/. Please go to the website, and download 'Digital_Music_5.json' and 'Kindle_Store_5.json' under the section titled "Small" subsets for expreimentation, i.e., the 5-core datasets.

## 2. Environment Setup

### 2.1 Anaconda Environment

1.  Open the Anaconda Prompt console and put the nlp.yaml in the same directory. Next put in the followinng command to create a anaconda environment named 'nlp'. then all packages needed will be installed automatically.

```
conda env create -f nlp.yaml
```

2.  Activate the 'nlp' environment with the following command.

```
conda activate nlp
```

### 2.2 Input dataset

1. Create a new directory named 'data' under the root directory of the project codes, then put 'Digital_Music_5.json' and 'Kindle_Store_5.json' into 'data' directory.

## 3. How to Run the project

### 3.1 Data Analysis
1. Open the Jupyter notebook "Data Analysis.ipynb"
2. Simply run each code cell in order from top to bottom. The first line in each cell explains the function
of this cell.

### 3.2 Simple Search Engine

1. Set up the system.

   ```
   python Search\ Engine.py
   ```

2. Input the query you want to search with the format of "reviwerID* asin* plain-text*" (order is interchangeable here and * represents 0 or more occurrences of the preceding term). Then press enter to confirm.
3. If you want to quit the system, simply type q and press enter to confirm.
4. The sample output will be a table with the searching results which has 6 columns: Rank, DocID, ReviewerID, asin, Snippets, and Score.

### 3.3 Review Summarizer

1. Open the Jupyter notebook "Recommender System (Collaborative Filtering System).ipynb".
2. Run the code cells from top to bottom.
3. You can adjust the number in the first [] of "sorted_processed_reviewText" in code block 10 and 11 to change products.
4. The output result of our summarizer is below code block 19.
5. The outputs of code blocks 12-16 are of baseline models. From the top to the bottom is TextRank, YAKE!, TfIdf, and TopicRank.

### 3.4 Application

1. Open the Jupyter notebook Recommender System (Collaborative Filtering System).ipynb.
2. Run the code block from top to bottom.
3. In the code block 19, the outputs show the sample results of a test SVD model and its RMSE value. In the code block 21, you can adjust the number in [] to get a product ID in the output, change the i in code block 22 according the obtained product ID, and run below code blocks, the top 10 recommended product will be shown in code block 24.
