#### Table of contents
- [ Introduction](#-introduction)
- [ Pipeline](#-pipeline)
- [ Approach](#-approach)
  - [Step 1: create index elasticsearch](#step-1-create-index-elasticsearch)
  - [Step 2: Get top k-articles ralated:](#step-2-get-top-k-articles-ralated)
  - [Step 3: Get raw answer](#step-3-get-raw-answer)
  - [Step 4: Post-process](#step-4-post-process)
- [ Preparing](#-preparing)
- [ Training](#-training)


# <a name="introduction"></a> Introduction

Due to the high quality of the MRC model, we only focus on the retrieval model which looking for the most relevant documents with the given question.

# <a name="pipeline"></a> Pipeline
<p align="center">
  <img src="./assets/images/qna-pipeline.png">
</p>

# <a name="approach"></a> Approach
## Step 1: create index elasticsearch
Wikidump will be pushed to elasticsearch, each document is a wikipage.

## Step 2: Get top k-articles ralated:
Tokenize the input query using VncoreNLP.
**Rule**:
- If the query input has proper noun such as Vua Nguyễn Huệ, Hà Nội, etc , we will query elasticsearch by these tokens and the document must has at least one of them.
- If the query input has not proper noun, we will search elasticsearch by these tokens and full query input.

And finally, top k-documents will be re-ranking by intersection-based score.

## Step 3: Get raw answer
Using MRC model to extract answer from top k-articles.

## Step 4: Post-process
Each question belongs to one of three categories : The question about wikipage, the question about date time, the question about number.
Based on its category to format answer.

# <a name="preparing"></a> Preparing
# <a name="training"></a> Training