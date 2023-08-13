# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
from glob import glob
def write_categories():

    dataset_name = 'ERA'  #
    with open('%s-labels.csv'% dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open('category.txt','w') as f:
        f.write('\n'.join(categories))


def write_video_txt(VIDEO_ROOT,flag,filename_output):
    video_all_path = sorted(os.listdir(os.path.join(VIDEO_ROOT,flag)),key=str.lower)
    print(video_all_path)
    output = []
    for i,video_path_index in enumerate(video_all_path):
        # print(os.path.join(VIDEO_ROOT,flag,video_path_index))
        video_paths = sorted(os.listdir(os.path.join(VIDEO_ROOT,flag,video_path_index)),key=str.lower)
        # print(video_paths)

        # print(video_paths)
        for j,video_path in enumerate(video_paths):

            frames_number = len(glob(os.path.join(VIDEO_ROOT,flag,video_path_index,video_path)+'/*'))
            output.append('%s %d %d' %("".join(os.path.join(VIDEO_ROOT,flag,video_path_index,video_path).split()),frames_number,i))

    print(output)
    print(filename_output)
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
    print(video_all_path)



# VIDEO_ROOT = '../dataset/UCF-crime_frames/'
# flag_train = 'train'
# flag_test = 'test'
# filename_output_train = '../dataset/UCF-crime_frames/train_split.txt'
# filename_output_test = '../dataset/UCF-crime_frames/test_split.txt'
#
#
# write_video_txt(VIDEO_ROOT,flag_train,filename_output_train)
# write_video_txt(VIDEO_ROOT,flag_test,filename_output_test)

def write_video_path(clip5s_frames_root,flag,output_path):

    all_paths = sorted(glob(clip5s_frames_root+'/'+flag+'/*'))

    output = []
    for i,clip_frame_root in enumerate(all_paths):
        print(clip_frame_root)
        clip_frame_paths = sorted(glob(clip_frame_root+'/*'))

        for j,clip_frame_path in enumerate(clip_frame_paths):
            print(clip_frame_path)
            frames_number = len(glob(clip_frame_path + '/*'))
            if i==0:
                kk=3
            if i==1:
                kk=6
            output.append('%s %d %d' % (clip_frame_path, frames_number, kk))

        # if i==0:
        #     break
    # print(output)
    with open(output_path,'w') as f:
        f.write('\n'.join(output))




def split_train_test():

    clip5s_frames_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames'
    flag_train = 'UCF12_crime'
    flag_test = 'test'
    train_output_path = clip5s_frames_root + '/train_split.txt'
    test_output_path = clip5s_frames_root + '/test_split.txt'

    write_video_txt(clip5s_frames_root,flag_train,train_output_path)

def add_fighting_test():
    clip5s_frames_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/RWF_frames'
    flag_train = 'test'
    train_output_path = clip5s_frames_root + '/test_split.txt'
    write_video_path(clip5s_frames_root, flag_train, train_output_path)


if __name__ == '__main__':
    split_train_test()
    # add_fighting_test()