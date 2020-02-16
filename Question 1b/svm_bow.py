from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, make_scorer
from sklearn.preprocessing import Normalizer

def make_pipeline():
    # Use CountVectorized as a BoW feature extractor
    # Use LinearSVC as an SVM
    pl = Pipeline([('vect', CountVectorizer()),('nrm', Normalizer()), ('svm', LinearSVC())])
    return pl

def evaluate(df, pl):

    scoring = { 'acc': 'accuracy',
                'pre': make_scorer(precision_score, average = 'weighted'),
                'rec': make_scorer(recall_score, average = 'weighted'),
                'f1': make_scorer(f1_score, average = 'weighted') }

    scores = cross_validate(pl, df['Clean_Data'], df['Label'], cv = 5, scoring=scoring, n_jobs=-1)
    
    test_acc = scores['test_acc'].mean()
    test_pre = scores['test_pre'].mean()
    test_rec = scores['test_rec'].mean()
    test_f1 = scores['test_f1'].mean()

    print("Accuracy: %0.5f" % test_acc)
    print("Precision: %0.5f" % test_pre)
    print("Recall: %0.5f" % test_rec)
    print("F-Measure: %0.5f" % test_f1)
    
    return test_acc, test_pre, test_rec, test_f1 

def fit(train_df, pl):
    pl.fit(train_df['Clean_Data', train_df['Label']])

def predict(test_df, pl):
    test_df['Predicted'] = pl.predict(test_df)
    return test_df

