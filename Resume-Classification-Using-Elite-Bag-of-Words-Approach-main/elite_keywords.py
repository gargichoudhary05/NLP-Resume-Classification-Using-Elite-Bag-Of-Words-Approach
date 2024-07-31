import pandas as pd
import numpy as np
import math

data = pd.read_csv('/content/train.csv')
data = data.dropna()
data = data.drop(['ID'], axis=1)
data = data.sort_values(by=['Category'],ascending=True)
data = data.reset_index(drop=True)

indices = []
finalResult = set()
categories = list(data.Category.unique())

for category in categories:
  st =  data.loc[data['Category'] == category].index[0]
  et =  data.loc[data['Category'] == category].index[-1] + 1
  indices.append([str(st), str(et), category])

indices

def getFinalEliteWords(st, et, category):
  acc = data.iloc[st:et]
 
  acc_token_index = {}
  counter = 0

  for sample in acc['Updated_resume']:
    for considered_word in sample.split():
      if considered_word not in acc_token_index:
        acc_token_index.update({considered_word : counter+1})
        counter = counter+1

  results  = np.zeros(shape = (len(acc['Updated_resume']),
                              max(acc_token_index.values()) + 1))

  for i, sample in enumerate(acc['Updated_resume']): 
    

    for j, considered_word in list(enumerate(sample.split())):
      

      index = acc_token_index.get(considered_word)
      
    
      results[i, index] = results[i,index]+1


  ##cummulative 
  row = len(results)
  col = len(results[0])
  acc_array = np.zeros(shape = (1,col))

  for i in range(0,row):
    for j in range(0,col):
      acc_array[0][j] = acc_array[0][j]+results[i][j]

  ##relative 
  acc_sigma = 0 
  for i in range(0,len(acc_array[0])):
    acc_sigma = acc_sigma + acc_array[0][i]

  for i in range(0,len(acc_array[0])):
    acc_array[0][i] = acc_array[0][i]/acc_sigma



  acc_word_list = []
  acc_word_list.append(' ')
  for key in acc_token_index:
    acc_word_list.append(key)

  acc_array_t = acc_array.T
 

  list_of_tuples = list(zip(acc_word_list, acc_array_t))
  acc_df = pd.DataFrame(list_of_tuples,columns=['words','prob'])

  

  acc_df = acc_df.sort_values(by = ['prob'],ascending=False)

  acc_df = acc_df.reset_index(drop=True)
  

  #mep
  max_H = 0
  max_k = -1

  for i in range(2,len(acc_df['prob'])-1):
    g_sum_1 = 0
    for j in range(0,i):
      g_sum_1 = g_sum_1 + acc_df.loc[j].at['prob'][0]
    g_sum_2 = 0
    for j in range(i,len(acc_df['prob'])):
      g_sum_2 = g_sum_2 + acc_df.loc[j].at['prob'][0]

    g1_arr = np.zeros(i)
    g2_arr = np.zeros(len(acc_df['prob'])-i)
    x = pow(10,-12)
    for j in range(0,i):
      g1_arr[j] = x+(acc_df.loc[j].at['prob'][0]/g_sum_1)

    count=0
    for j in range(i,len(acc_df['prob'])):
      g2_arr[count] = x+(acc_df.loc[j].at['prob'][0]/g_sum_2)
      count = count+1

    H1 = 0
    for j in range(0,i):
      H1 = H1+(g1_arr[j]*math.log(g1_arr[j],2))

    H2 = 0
    for j in range(0,len(acc_df['prob'])-i):
      H2 = H2+(g2_arr[j]*math.log(g2_arr[j],2))

    H = -(H1+H2)

    if max_k==-1:
      max_H = H
      max_k = i

    else :
      max_H = max(max_H,H)
      if max_H==H:
        max_k = i


  print("No of elite keywords in category {} : {}".format(category, max_k))
  result = acc_df['words'].iloc[:max_k].tolist()
  finalResult.update(result)

for idx in indices:
  getFinalEliteWords(int(idx[0]), int(idx[1]), idx[2])

print(finalResult, len(finalResult))