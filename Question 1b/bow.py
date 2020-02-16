import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd

def get_bow(data_frame):

    # Grouping by Label
    gb = data_frame.groupby(['Label'])
    
    # Creating a dictionary of the form {key: Label, value: Data(Title+Content lines)}
    docs = {x: ' '.join(gb.get_group(x).drop(columns=['Label'])['Clean_Data'].tolist()) for x in gb.groups}

    # Order the dictionary to ensure the order of the keys matches the order of the values when accessed separately
    # in order to add the labels in the correct order in the dataframe
    docs = collections.OrderedDict(docs)

    # Initialize a vectorizee that will tokenize our data, and produce frequencies of all tokens per document (category)
    cv = CountVectorizer()

    # Get the frequency vector. (Perform tokenization and calculate token frequencies)
    vector = cv.fit_transform(docs.values())

    # Convert the vector to array
    vector_arr = vector.toarray()

    # Generate a pandas dataframe with features as columns and document names (categories) as row indexes
    bow_df = pd.DataFrame(vector_arr, columns=cv.get_feature_names(), index=docs.keys())
    return bow_df