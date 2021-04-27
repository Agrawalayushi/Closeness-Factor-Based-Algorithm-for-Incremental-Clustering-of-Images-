#importing libraries
import pandas as pd
import math as mp
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# independent function
#function to represent any incremental dataset
def visualize_increment(df_ic):
  ax = df_ic.groupby(['CNumber'])['CNumber'].count().plot.bar(title = " Recent Incremental clustering")
  ax.set_xlabel('Clusters')
  ax.set_ylabel('Frequency')
  plt.show()


#to visualize dataset before the addition of new dataset(increment)
def visualize_basic():
  df_basic = pd.read_csv('record.csv')
  ax = df_basic.groupby(['CNumber'])['CNumber'].count().plot.bar(title = " Basic clusters Formed")
  ax.set_xlabel('Clusters')
  ax.set_ylabel('Frequency')
  plt.show()

# merging basic and incremental dataset
def mergefile_graph(df_basic,df_incremental):
  df_basic['Cluster_Type'] = 'Basic_cluster'
  df_incremental['Cluster_Type'] = 'Incremental_1'
  df_basic = df_basic.append(df_incremental)
  df_basic=df_basic.sort_values(by = ['CNumber'])
  df_basic.to_csv('record.csv',index = False)
  print("df_basic length", len(df_basic))
  return df_basic


#visulize different increments
def visualize_all(df_merger):
  ax = df_merger.groupby(['CNumber','Cluster_Type'])['Cluster_Type'].count().unstack(0).plot.bar(stacked=True, figsize=(8, 6))
  ax.legend(loc = 'center right',bbox_to_anchor = (1.4,0.5),ncol = 1)
  plt.title('Incremental Iteration')
  plt.xlabel('clusters')
  plt.ylabel('No of Records')
  plt.show()

#incremental clustering code
def incremental_cluster(df_ic):
  df = pd.read_csv('record.csv')
  df_rep = df
  print("test data",df.head())
  
  print(df_rep.head())
  df_rep['row_total'] = df_rep.sum(axis =1)
  whole = []
  outlier = []
  fclose=[]
  outlierclose=[]
  print(df_ic)
  df_ic['row_total'] = df_ic.sum(axis =1)
  for i in range(len(df_ic)):
    df_ic.loc[i,'Flag']=False
  c1 = []
  for i in range(len(df_rep)):
    whole.append(i)
    print("whole",whole)
    for j in range(len(df_ic)):
      if(df_ic.Flag[j]==False):
        c1 = df_rep.row_total[i]/(df_rep.row_total[i]+df_ic.row_total[j])
        
        d1 = df_rep.T1[i]+df_ic.T1[j]
        d2=c1*d1-df_rep.T1[i]
        d3 = mp.sqrt(d1*c1*(1-c1))
        prob1 = d2/d3
        c_square = mp.pow(prob1,2)
        weight = mp.sqrt(d1)
        c = c_square * weight
       
        #second feature - V1
        col2 = df_rep.V1[i]+df_ic.V1[j]
        col21 = (c1*col2-df_rep.V1[i])/mp.sqrt(col2*c1*(1-c1))
        e2 = mp.pow(col21,2)
        wei2 = mp.sqrt(col2)
        c2 = e2 * wei2
        

        #fourth feature W1
        col4 = df_rep.W1[i]+df_ic.W1[j]
        col41 = (c1*col4-df_rep.W1[i])/mp.sqrt(col4*c1*(1-c1))
        e4 = mp.pow(col41,2)
        wei4 = mp.sqrt(col4)
        c4 = e4 * wei4


        close1 = c+c2+c4
        close2 = weight+wei2+wei4
        close = close1/close2

        if close<=1:
          whole.append(j)
          df_ic.loc[j,'Flag']=True
          df_ic.loc[j,'CNumber']=df.CNumber[i]
          
        else:
          outlier.append(j)
          outlierclose.append(close)
    fclose.append(0)

  resultant_list = list(set(outlier)-set(whole))
  print('Difference is :',resultant_list)
  if(len(resultant_list)!=None):
    df_ic.loc[resultant_list,'CNumber']=i+2
    df_ic.loc[resultant_list,'Flag']=True
    
  df_ic =df_ic.drop(['Flag','row_total'],axis=1)
  return df_ic,df

def scale(pandas_df):
  features = ['T1','V1','W1']
  features_v = pandas_df[features]
  scaler = MinMaxScaler(feature_range = (0,10))
  scaler_features = scaler.fit_transform(features_v)
  print("normalised dataset with MinMaxScaler",scaler_features)
  df2 = pd.DataFrame(scaler_features,columns = ['T1','V1','W1'])
  return df2


#main function
def main():
  start = time.time()
  read_file_path = input('Enter the Path of the File : ')
  df = pd.read_csv(read_file_path)
  #Store in DataFrame
  print("Dataframe Loaded...........")
  scaler_features = scale(df)
  print("Scaled", scaler_features)
  print("Dataframe normalised")
  print("Visualise the Basic clusters....")
  visualize_basic()
  print("Incremental Clustering Processing...........")
  df_incremental, df_og = incremental_cluster(scaler_features)
  print("Incremental Clustering Processed")
  visualize_increment(df_incremental)
  df_merger = mergefile_graph(df_og, df_incremental)
  print("Visualise All increments")
  visualize_all(df_merger)
  print('Execution time {} seconds'.format(time.time()-start))

if __name__=='__main__':
  main()