import pandas as pd
import string
df = pd.read_csv('Alexa-Dataset.csv')
#graph-positive-negavtive
feedback_counts = alex['feedback'].value_counts()
plt.bar(feedback_counts.index, feedback_counts.values)
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.title('Distribution of Positive and Negavtive Feedbacks')
plt.xticks([0, 1], ['Negavtive', 'Positive'])
plt.show()
#lowercase
df["verified_reviews"] = df["verified_reviews"].str.lower()
#remove punctuations
punc = set(string.punctuation)
def removePunc(Text):
    Text = str(Text)
    ret = ''
    for i in Text:
        if i not in punc:
            ret += i
    return ret
removePunc("abcd!@#$")
df["verified_reviews"].apply(removePunc)
df["verified_reviews"] = df["verified_reviews"].apply(removePunc)
#Tokenize
from nltk.tokenize import word_tokenize
word_tokenize("fghfg fdgdfg dfgsdfg")
df["verified_reviews"].apply(word_tokenize)
df["verified_reviews"] = df["verified_reviews"].apply(word_tokenize)
#remove stopwords
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
df["verified_reviews"].apply(lambda x: [word for word in x if word not in sw])
df["verified_reviews"] = df["verified_reviews"].apply(lambda x: [word for word in x if word not in sw])
#stemming
from nltk.stem import PorterStemmer
stemer = PorterStemmer()
stemer.stem("eating")
df["steamed_reviews"] = df["verified_reviews"].apply(lambda x: [stemer.stem(word) for word in x] )
#Lemmatization
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
wnl.lemmatize("playing", pos="v")
df["lemmatized_review"] = df["verified_reviews"].apply(lambda x: [wnl.lemmatize(word, pos="v") for word in x])
#Perform the word vectorization on review text using Bag of Words technique
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(df['lemmatized_review'].apply(lambda x: ' '.join(x)))
bow_matrix.data
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
bow_df.zero.sum()
#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatized_review'].apply(lambda x: ' '.join(x)))
tfidf_matrix