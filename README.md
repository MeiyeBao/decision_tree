
# Decision Tree

## Project Description: Customer Segmentation
The target of this project is to classify customers' purchasing behaviour based on customers' basic information.


## Data Cleaning 

1. Age: <=30: 1 , [31:40] : 2, >40: 3
2. Incoming: low: 1 , medium: 2 , high: 3 3. Student: no: 0 , yes: 1
4. Credit Rating: fair: 1 excellent: 2
5. Buying: no: 0 , yes: 1


## Stratified Test-Train Split
Understanding the purpose of using the decision tree in this classification question, stratified test-train split is one of the best choices in classification questions with small sample sizes.

A split of 80% training set and 20% test set is used in the code.


## Ways to Split Decision Tree
To obtain the greatest information gain, the gini index and entropy are used. The gini index returns better accuracy.

$$ I_G = 1 - \sum^c_{i=1} p_i^2$$

## Result

The final trained decision tree was formed using the Gini Index as below:


<img width="589" alt="Screen Shot 2023-02-18 at 4 57 24 PM" src="https://user-images.githubusercontent.com/123518900/219851592-75d2c035-d29e-4419-bf16-927f39e2ac93.png">
