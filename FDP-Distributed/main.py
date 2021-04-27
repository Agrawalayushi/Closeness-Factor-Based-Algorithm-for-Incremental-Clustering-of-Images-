import pandas as pd
from Datapreprocess import *
from cfba import *

import time


def main():
  start = time.time()
  df = pd.read_csv(r'C:\Users\pc\Desktop\College\BTECH\FDP-Distributed\Dataset\record.csv')
  df = cleanData(df)
  df_copy = df.copy()
  df = encodeData(df)
  print(df.head())
  train,test,df1,df2 = scale(df)
  df_basic,df_train = basic_cluster_lone(train,df1)
  df_incremental , dftest = incremental_cluster(test,df2)
  merge_basic_incremental1 = mergefile_graph(df_basic,df_incremental)
  mergefile_representative(df_train,dftest)
  visualize(df_basic,df_incremental,merge_basic_incremental1)
  print('Execution time {} seconds'.format(time.time()-start))

if __name__=='__main__':
  main()
