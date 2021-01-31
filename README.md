# Keypoints classifications

![alt](https://github.com/FrancescoSaverioZuppichini/scientists-keypoints-classification/blob/develop/images/how_i_did_it.jpg?raw=true)

## Problem
The problem was to classify 5 keypoints classes

![alt](https://github.com/FrancescoSaverioZuppichini/scientists-keypoints-classification/blob/develop/images/problem.png?raw=true)

## Data
Each `.csv` file contains a sequence of keypoints, there are 18 x,y pairs of keypoints per time step, $t$. At each time $t$ `n` keypoints may be missing.

### Preprocessing

I add `0`s in the index of the missing keypoints, and I shift them to the top left corner

### Normalization

Each keypoint is normalized using the whole training $mean$ and $std$.

## Models

I treat the problem as sequence classification. I used 1D convolution to classify each sequence. I choose an empirical sequence size, $S$ of $9$. 

For completeness, I also classify each row individually (so $S = 1$) using a dense network.

The results are shown in the following screenshot 

![alt](https://github.com/FrancescoSaverioZuppichini/scientists-keypoints-classification/blob/develop/images/logs.png?raw=true)

**I did not have a lot of time, just a few hours in the weekend.** So I just evaluate a couple of models keeping always the same seed and the best performing one is a conv1d model with three layers of increasing width (`32, 64, 128`). Nothing fancy here.

## Evaluation
To get the final prediction for each test file I just classify each sequence $S_i, \text{where} \ 0 < i < |Dataset|$ for each dataset using the best performing model. We use a sliding window of 1. We then vote for the most frequent class and make the final prediction. 

## Conclusion

I didn't have a lot of time, If I had more this is the thing I would have loved to do

- data exploration
  - are there any correlations? E.g. some classes occupy more space?
  - who is the most frequent one?
  - who is the class that can be more confused?
- data preparation
  - make the windows size not fixed to 1, this does not make a lot of sense since lots of keypoints will just appear in the next sequence
- result analysis
  - confusion matrix
  - plots
  

Thanks!