## Best model Pre Trained with the tweet dataset ##

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB
import emotion_detection


def getModel():
    #Creating Tfidf Vectorizer from our dataset
    train, y, tfidf = emotion_detection.initialize()
    #Select K Best Feature Selection Model based on Best features obtained from experimentation
    X_new = SelectKBest(chi2, k=900)
    new = X_new.fit_transform(train, y)
    #Applying Best Model with Best parameters after experimenting with multiple models
    naive_bayes = BernoulliNB(fit_prior=False, alpha=10.0, binarize=0.0)
    naive_bayes.fit(new, y)
    return naive_bayes, tfidf, train, y, X_new

