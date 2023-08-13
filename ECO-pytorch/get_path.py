import os
import random
from glob import glob
'''
train_out = open("./test_split_all.txt",'w')
#test_out = open("./test_split_all.txt",'w')
train_lines=[]
#test_lines = []
with open("./test_split.txt",'r') as infile:
  for i,line in enumerate(infile):
    print(i,line)
    line = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/ERA/'+ line[11:]
    print(i,line)
    train_lines.append(line)
    
  #random.shuffle(train_lines)
  #print(train_lines)

  for train_line in train_lines:
    train_out.write(train_line)
  
'''
train_out = open("./train_split_07.txt",'w')
test_out = open("./test_split_03.txt",'w')
train_lines = []
test_lines = []
data_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/UCF12_crime'
train_id = [[1,390,558,627],[1,17,23,40],[1,137,202,352],[1,559,806,1365,1606,1745],[1,80,115,1516],[1,12],[1,2407,3651,3860,3927,4091,4171,4304,4363,4470],[1,553,792,808,817,834],[1,233,334,372,390,426],[1,175],[1,169],[1,74,105,224]]
print(train_id[0])
test_id = [[391,557,628,657],[18,22,41,46],[138,201,353,418],[560,805,1366,1605,1746,1805],[81,114,1517,2127],[13,16],[2408,3650,3861,3926,4092,4170,4305,4362,4471,4517],[554,791,809,816,835,841],[234,333,373,389,427,442],[176,250],[170,243],[75,104,225,276]]
video_labels = sorted(glob(data_root+'/*'))
for i,video_label in enumerate(video_labels):
   videos = sorted(glob(video_label+'/*'))
   for j,video in enumerate(videos):
      
      frames = sorted(glob(video+'/*.jpg'))
      frame_number = len(frames)
      #print(video,frame_number,i)
      train_flag = False
      test_flag = False
      for k in range(len(train_id[i])):
        if(k%2==0):
      	   if((train_id[i][k]-1)<=j<=(train_id[i][k+1]-1)):
             train_flag = True
           
      for k in range(len(test_id[i])):
        if(k%2==0):
      	   if((test_id[i][k]-1)<=j<=(test_id[i][k+1]-1)):
             test_flag = True
      if(train_flag):
         train_lines.append(video + " " + str(frame_number) + " " + str(i) + "\n")
      if(test_flag):
         
         test_lines.append(video + " " + str(frame_number) + " " + str(i) + "\n")
print(len(train_lines))
print(len(test_lines))
for i,line in enumerate(train_lines):
    train_out.write(line)

for j,line in enumerate(test_lines):
    test_out.write(line)

