
#### first read train.json
import json 
from googletrans import Translator



f = open("trainset.json")
data=json.load(f)

# print("data in json file: ", data)
# print("type of data in json file: ", type(data)) ### json file is list of dictionary 



print("number of item in original json file: ",len(data))
# exit()

translator = Translator()

counter=0
new_data=[]
for i in range(0,len(data)):

    # print(type(data[i]["sent"]))
    # print(data[i])
    # exit()
    try:
        traslated_sentance= translator.translate(data[i]["sent"],src="en",dest="fr").text
        # print(traslated_sentance)
        traslated_back_sentence= translator.translate(traslated_sentance,src="fr",dest="en").text
    

        if data[i]["sent"]!=traslated_back_sentence:

            # new_dict=data[i]
            # new_dict["sent"]=traslated_back_sentence
            temp=data[i].copy()
            new_data.append(
                {
                    'answer_type':temp['answer_type'],
                    'img_id':temp['img_id'],
                    'label':temp['label'],
                    'question_id':temp['question_id'],
                    'question_type':temp['question_type'],
                    'sent':traslated_back_sentence,


                }
            )
        
        # print("the original sentence: ",data[i]["sent"])
        # print("the translated sentence: ", traslated_back_sentence)

        if i%500==0 and i!=0:
            data.extend(new_data)
            new_data=[]
            with open('data/train_augmented.json', 'w') as outfile:
                json.dump(data, outfile)
                
            print(len(data))
        

    except AttributeError:
        print("can not do {}".format(data[i]["sent"]))
    
    
    # print("the original sentence: ",data[i]["sent"])
    # print("the translated sentence: ", traslated_back_sentence)
    # if i==10:
    #     exit()
    



