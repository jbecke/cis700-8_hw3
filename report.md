# Homework 3: Natural Language Understanding

## Sentiment Classifier
### Basic Sentiment Classifier
Using the most positive candidate (according to TextBlob) yields the following accuracies:

```
2016 validation accuracy: 
0.5777659005879209

2018 validation accuracy: 
0.558879694462126

2016 test accuracy: 
0.5740245857830037
```

### Slightly better classifier
For the next part, we tried to see if we could use context to improve our sentiment-based classifier.

First, we tried decision code based on the difference in sentiment of the final sentence vs. the context sentiment. Context sentiment is the unweighted average of the sentiment of the context sentences. Our intuition was that if the context sentiment was negative, the final sentiment should be negative and vise-versa. Unfortunetly, this got lower than 50% accuracy.... ok so maybe the intuition should be flipped; maybe there is some "shock value" at the end of stories that flips sentiment.

We also tried using the sentiment of the 4th sentence instead of the average context sentiment, but this yielded little change.


We then tried to combine this model with the origional sentiment model. The decision code for this part looks like this:

```python
if sentiment1 - sentiment2 > .1:
  predictions.append(0)
elif sentiment2 - sentiment1 > .1:
  predictions.append(1)
elif delta1 > delta2:
  predictions.append(0)
else:
  predictions.append(1)
```

Where `delta1` and `delta2` are the differences between the respective 5th sentences and the average context sentiment. This gave very slightly better results, probably not significant:

```
2018 validation accuracy: 
0.5824315722469765

2016 test accuracy: 
0.5702832709780866
```

### Error analysis
Context:
```
['My son has always been very bright.', 'However, he is not very good at doing his homework.', 'Last trimester, he almost failed three of his classes.', "He had to work very hard to get his grades back up by term's end."]

```
Analysis:
```
Prediction He was a great student all the time.
Option 0 But after working hard, he managed to pass.
Option 1 He was a great student all the time.
Sentiment of 0 -0.2916666666666667
Sentiment of 1 0.8
```

This is a very obvious case that shows how the model can fail. Obviously, based on the context, Option 1 is false. However, due to the model only relying on the positive sentiment, this is its prediction.

## BERT Classifier
Using the vanilla MLP with a ReLU non-linearity, we were able to yield the following accuracies:

```
2016 validation accuracy: 
0.6472474612506681

2018 validation accuracy: 
0.6543602800763845

2016 test accuracy: 
0.6563335114911811
```

Adding an extra layer, with a size of 128 and ReLU activation gave the following results:

```
2016 validation accuracy: 
0.5531801175841796

2018 validation accuracy: 
0.5493316359007002

2016 test accuracy: 
0.55264564404062
```

This didn't make sense to us, since this is deep learning and more layers equals more performance (/s). So, we changed the learning rate from 0.001 to 0.002 and doubled the number of training steps to 20,000. Our intuition was that perhaps the network wasn't training for long enough, and because we have a due date for this HWK and needed to achieve results, running the model for many times longer wasn't an option. We thought a combination of increasing the learning rate and more training steps would be a compromise. This yielded the following accuracies:

```
2016 validation accuracy: 
0.6419027258150721

2018 validation accuracy: 
0.642266072565245

2016 test accuracy: 
0.6451095670764297
```

Which was closer to the original vanilla MLP. Finally, we decided to add Dropout after the input layer to prevent any sort of overfitting. In respose, we also increased the learning rate and number of steps to yield:

```
2016 validation accuracy: 
0.649919828968466

2018 validation accuracy: 
0.6499045194143858

2016 test accuracy: 
0.6557990379476216
```

Which yielded comparable results. We hypothesize that the model would improve with more training time.

### *Cheating* with the Validation Set
When we train using the validation set with the included vanilla MLP code, we yield the following accuracies:

```
2016 test accuracy: 
0.69000534473543561
```

Increasing the first layer to have a size of 1000 yielded a small accuracy improvement:

```
2016 test accuracy: 
0.7022982362373063
```

Increasing the number of number of training examples from validation set by 1000, and training for 30,000 steps yields the following accuracies:

```
0.7183324425440941
```

Finally, adding another dense layer of size 128 with ReLU activation gives us the following results:

```
2016 test accuracy: 
0.7017637626937466
```
