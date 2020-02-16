import pandas as pd
import os.path
from bow import get_bow
from data_pre_processing import pre_process
import svm_bow

def init_train():    
    if os.path.isfile('c:\\Users\\Konio\\Desktop\\asssignment_big_data\\train_clean.csv'):
        df = pd.read_csv('C:\\Users\\Konio\\Desktop\\asssignment_big_data\\train_clean.csv')
        df = df.drop(columns=['Unnamed: 0'])
    else:
        df = pd.read_csv('C:\\Users\\Konio\\Desktop\\datasets\\q1\\train.csv')

        # Joining Title and Content columns
        df['Data'] = df['Title'].astype(str) + ' ' + df['Content']
        
        # Droping unwanted columns 
        df = df.drop(columns=['Id', 'Title', 'Content'])

        df = pre_process(df)

        df.to_csv('c:\\Users\\Konio\\Desktop\\asssignment_big_data\\train_clean.csv')

    return df

def init_test():    
    if os.path.isfile('c:\\Users\\Konio\\Desktop\\asssignment_big_data\\test_clean.csv'):
        test_df = pd.read_csv('C:\\Users\\Konio\\Desktop\\asssignment_big_data\\test_clean.csv')
    else:
        test_df = pd.read_csv('C:\\Users\\Konio\\Desktop\\datasets\\q1\\test_without_labels.csv')

        # Joining Title and Content columns
        test_df['Data'] = test_df['Title'].astype(str) + ' ' + test_df['Content']
        
        # Droping unwanted columns 
        test_df = test_df.drop(columns=['Id', 'Title', 'Content'])

        test_df = pre_process(test_df)

        test_df.to_csv('c:\\Users\\Konio\\Desktop\\asssignment_big_data\\test_clean.csv')

    return test_df

if __name__ == "__main__":
    df = init_train()
    test_df = init_test()

    df = df.drop(columns=['Data'])

    sb_pl = svm_bow.make_pipeline()
    # performing evaluation for svm using BoW
    sb_test_acc, sb_test_pre, sb_test_rec, sb_test_f1 = svm_bow.evaluate(df, sb_pl)
    svm_bow.fit(df, sb_pl)
    svm_bow.predict(test_df, sb_pl)
    
     

    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    pass


