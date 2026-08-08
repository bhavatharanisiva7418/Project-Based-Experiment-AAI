
# Project Based Experiment<H1>

#### NAME: BHAVATHRANI S
#### REGISTER NO :212223230032
#### DATE: 8/8/26


 ## Objective:
 To perform sentiment analysis on Facebook posts using the VADER sentiment analyzer and filter the posts that contain negative feedback.

## Program:
```
!pip install pandas nltk

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Sample Facebook posts
data = {
    'Timeline': [
        'I really enjoyed this amazing event!',
        'The service was excellent and very helpful.',
        'I love this new update!',
        'I am very disappointed with the service.',
        'This was a terrible experience.',
        'The event was okay.',
        'The product quality is very bad.'
    ]
}

df = pd.DataFrame(data)

sid = SentimentIntensityAnalyzer()

df['Compound'] = df['Timeline'].apply(
    lambda x: sid.polarity_scores(x)['compound']
)

negative_feedback = df[df['Compound'] < 0]

print("Negative Feedback:")
print(negative_feedback[['Timeline', 'Compound']])
```

## Output:
<img width="1482" height="385" alt="image" src="https://github.com/user-attachments/assets/c121fb24-f951-464c-80f3-d302ce3aa949" />


## Inference:
The program successfully analyzed the sentiment of the given Facebook posts using VADER and calculated their compound sentiment scores. Posts with a compound score less than 0 were identified and filtered as negative feedback. Thus, sentiment analysis can be used to identify negative opinions from social media data.
