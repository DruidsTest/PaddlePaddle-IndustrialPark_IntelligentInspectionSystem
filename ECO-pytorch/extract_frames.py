
import threading
import cv2
import os
import sys
from glob import glob




def dump_frames(vid_path,out_path,video_flag):

    video = cv2.VideoCapture(vid_path)
    vid_name = os.path.basename(vid_path)
    # vid_name = out_path.split('/')[-1]
    print(vid_name)

    try:
        os.mkdir(out_path)
    except OSError:
        pass
    file_list = []
    i=1
    while(True):

        ret, frame = video.read()
        # print(frame)
        if ret is False:
            break
        # print('{}/{:06d}.jpg'.format(out_full_path, i))
        cv2.imwrite('{}/{:05d}.jpg'.format(out_path, i), frame)
        access_path = '{}/{:05d}.jpg'.format(vid_name, i)
        i=i+1
        file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()
    return file_list



def extract_frame(abnormal_video):

    NUM_THREADS = 20
    VIDEO_ROOT = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut_5s/for4/' + abnormal_video
    FRAME_ROOT = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/for4/' + abnormal_video  # Directory for extracted frames

    try:
        os.mkdir(FRAME_ROOT)
    except OSError:
        pass

    cut5s_clips = sorted(glob(VIDEO_ROOT+'/*.mp4'))
    flag_video = 3651
    for i,cut5s_clip in enumerate(cut5s_clips):

        clip_basename = os.path.basename(cut5s_clip)
        # frames_path = FRAME_ROOT + '/' + clip_basename.split('.')[0]
        frames_path = FRAME_ROOT + '/' + abnormal_video + '_' +str(flag_video).zfill(5)
        print(frames_path)
        dump_frames(cut5s_clip, frames_path, flag_video)
        flag_video += 1
        # if i==0:
        #     break


if __name__ == '__main__':


    # abnormal_video = 'Escape'
    labels = ['Normal']

    for abnormal_video in labels:
        extract_frame(abnormal_video)



