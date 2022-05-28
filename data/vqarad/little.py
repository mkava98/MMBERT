import json
from sklearn.model_selection import train_test_split
# Opening JSON file
f = open('data.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
for i in data:
    print(i)
    break
_, test = train_test_split(data, test_size=0.2)
train, test = train_test_split(test, test_size=0.3)

with open('testlittle.json', 'w') as outfile:
    json.dump(test, outfile)
  
# Using a JSON string
with open('trainlittle.json', 'w') as outfile:
    json.dump(train,outfile)
# Closing file
f.close()