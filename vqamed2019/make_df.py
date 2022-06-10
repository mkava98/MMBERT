import pandas as pd 
import os 




def make_df(file_path, nameoffile):
    # paths = os.listdir(file_path)
    if (nameoffile != "testdf.csv"):

    # df_list = []
    
    # for p in paths:
        df = pd.read_csv(file_path, sep='|', names = ['img_id', 'question', 'answer'])
        # print(p)
        # df['category'] = file_path.split('_')[1]
        # print(df['category'])
        listt =file_path.split('_')
        df['mode'] = file_path.split('_')[3].split(".")[0]
        # print(df['mode'])

        # print(df['mode'])
        # df_list.append(df)
        # print(df)
        df.to_csv(nameoffile)
    else :
        df = pd.read_csv(file_path, sep='|', names = ['img_id', 'question'])
        listt =file_path.split('_')
        # print(listt)
        df['mode'] = file_path.split('_')[1]
        # print(df['mode'])
        df.to_csv(nameoffile)


        

    return df



def make_dfcat(file_path , nameoffile):
    paths = os.listdir(file_path)
    
    df_list = []
    
    for p in paths:
        df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
        df['category'] = p.split('_')[1]
        df['mode'] = p.split('_')[2][:-4]
        df_list.append(df)
    pd.concat(df_list).to_csv(nameoffile)
    return pd.concat(df_list)



def make_dftest(file_path, nameoffile):
    # paths = os.listdir(file_path)

    # df_list = []
    
    # for p in paths:
    df = pd.read_csv(file_path, sep='|', names = ['img_id','category', 'question', 'answer'])
    # print(p)
    # df['category'] = file_path.split('_')[1]
    # print(df['category'])
    listt =file_path.split('_')
    ## if we need this 
    ##df = df.reindex(columns=column_names)

    print(listt)
    df['mode'] = file_path.split('_')[1].lower()
    # print(df['mode'])

    # print(df['mode'])
    # df_list.append(df)
    # print(df)
    df.to_csv(nameoffile)


# make_df("ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt", "traindf.csv")
# make_df("ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt", "valdf.csv")

## make dataframe for testing
# make_df("ImageClef-2019-VQA-Med-Test/VQAMed2019_Test_Questions.txt", "testdf.csv")

make_dfcat("ImageClef-2019-VQA-Med-Training/QAPairsByCategory", "traindf.csv")
make_dfcat("ImageClef-2019-VQA-Med-Validation/QAPairsByCategory", "valdf.csv")

## make dataframe for testing
 
make_dftest("ImageClef-2019-VQA-Med-Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt", "testdf.csv")

