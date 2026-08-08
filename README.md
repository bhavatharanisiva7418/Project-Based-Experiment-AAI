<H3>ENTER YOUR NAME</H3>
<H3>ENTER YOUR REGISTER NO.</H3>
<H3>DATE:</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>
Type your objective based on the question
<H3>Program:</H3>
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
<H3>Output:</H3>
Show your execution results here
<H3>Inference:</H3>
Write about your learning experience out of this project. (What you have learned)
