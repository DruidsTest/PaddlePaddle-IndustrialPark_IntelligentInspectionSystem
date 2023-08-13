
import numpy as np
from glob import glob
import cv2
import os
import shutil

def cut_video_clip():

    video_data_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_total/for4'
    ann_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_ann/for4'
    video_cut_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut/for4'
    other_video_cut_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut/other_video'
    event_paths = sorted(glob(video_data_root+'/*'))
    ann_paths = sorted(glob(ann_root+'/*'))
    save_paths = sorted(glob(video_cut_root+'/*'))
    # print(event_paths,ann_paths)
    other_label_video = []
    labels = [12,9]
    for i,event_name in enumerate(event_paths):
        video_names = sorted(glob(event_name+'/*.mp4'))
        ann_names = sorted(glob(ann_paths[i]+'/*.txt'))
        save_path = save_paths[i]
        assert len(video_names)==len(ann_names)
        for j,ann_name in enumerate(ann_names):

            video_name = video_names[j]
            # print(video_name,ann_name)
            assert video_name.split('/')[-1].split('.')[0] == ann_name.split('/')[-1].split('.')[0]
            anns = np.loadtxt(ann_name).reshape(-1,4)
            # print(anns)
            if not anns.all():
                print(ann_name)
            # print(anns.shape)



            for k in range(anns.shape[0]):

                clip_label = anns[k][3]

                video_basename = os.path.basename(video_name)
                if clip_label != labels[i]:
                    other_label_video.append(video_basename)
                    print('error video label----------------------------------------------')
                    print(int(clip_label))
                    print(other_label_video)
                    save_clip_name = other_video_cut_root + '/' + str(int(clip_label)) + '/' + video_basename.split('.')[0] + '_{}'.format(k+1)+'.mp4'
                    print(save_clip_name)
                    print('___________________________________________________')
                    # print(len(other_label_video))
                else:
                    print(video_basename)
                    save_clip_name = save_path + '/' + video_basename.split('.')[0] + '_{}'.format(k+1)+'.mp4'
                    print(save_clip_name)

                ## fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                cap = cv2.VideoCapture(video_name)

                frames_fps = int(cap.get(5))
                img_width = int(cap.get(3))
                img_height = int(cap.get(4))
                img_size=(img_width,img_height)
                video_writer = cv2.VideoWriter(save_clip_name, fourcc, 30, img_size)
                # print("frames_fps:{}".format(frames_fps))
                frame_count = 0
                start_frame = int(anns[k][1] * frames_fps)
                end_frame = int(anns[k][2] * frames_fps)
                print(start_frame,end_frame)
                frame_diff = end_frame-start_frame
                if frame_diff<120:
                    end_frame = end_frame + (120-frame_diff) + 5
                success = True

                while (success):
                    success, frame = cap.read()
                    if frame_count>=start_frame and frame_count<end_frame:
                        # print('kkkkkkkkkkkk')
                        video_writer.write(frame)
                        # imgs.append(frame)
                    frame_count += 1

                video_writer.release()
        #     if j==0:
        #         break
        #
        #
        # if i==0:
        #     break


def change_ann():


    ann_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_ann/for3'
    ann_paths = sorted(glob(ann_root + '/*'))

    for i, ann_name in enumerate(ann_paths):
        ann_names = sorted(glob(ann_name + '/*.txt'))
        # print(ann_names)
        if  i>=0:
            for j, ann_name in enumerate(ann_names):

                txt_f = open(ann_name)
                lines = txt_f.readlines()
                true_lines = []
                for k in range(len(lines)):
                    if k>0:
                        true_lines.append(lines[k][1:])
                    else:
                        true_lines.append(lines[k])
                # print(true_lines)
                txt_f.close()
                # print(ann_name)
                write_f = open(ann_name,"w")


                write_f.writelines(true_lines)
                write_f.close()
            # while line:
            #     context = txt_f.readline()
            #     print(context)


            # if j==0:
            #     break


        # if i==0:
        #     break


def change_filename():
    ann_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut_5s/all/ParadeProtest'
    txt_names = sorted(glob(ann_root + '/P*.mp4'))

    for i,txt_name in enumerate(txt_names):
        # print(txt_name)

        new_txt_name = ann_root + '/' + 'Demonstration_' +str(i+52).zfill(5)+'.mp4'
        # print(new_txt_name)
        os.rename(txt_name,new_txt_name)

        # if i==0:
        #     break

def transform_videotime(abnormal_name):

    # abnormal_name = 'Demonstration'
    clip_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut/for4/' + abnormal_name
    short_clip_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut_5s/all/clip_lower5s'
    clip_5s_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut_5s/for4/' + abnormal_name

    clip_names = sorted(glob(clip_root + '/*.mp4'))
    clip_5s_fps = 24
    clip_5s_frames = 120
    start_clip5s_id = 3860
    for i,clip_name in enumerate(clip_names):
        if i<4: continue
        clip_basename = os.path.basename(clip_name)
        print(clip_basename)

        cap = cv2.VideoCapture(clip_name)
        # print(cap.get(5))
        frames_fps = int(cap.get(5))                            #得到帧率
        frames_num = int(cap.get(7))                            #得到总帧数
        img_width = int(cap.get(3))                             #图像宽度
        img_height = int(cap.get(4))                            #图像高度
        clip_time = frames_num / frames_fps
        img_size = (img_width, img_height)

        print('clip_time',clip_time,'frames_fps:',frames_fps,'img_width:',img_width,'img_height:',img_height)

        #判断是否可裁剪成一段视频
        if clip_time < 4:
            print(clip_basename,clip_time)
            short_clip_oldpath = clip_name
            short_clip_newpath = short_clip_root + '/' + clip_basename
            print(short_clip_oldpath,short_clip_newpath)
            shutil.move(short_clip_oldpath,short_clip_newpath)

        else:

            # print(clip_basename)
            # print('clip_time', clip_time, 'frames_fps:', frames_fps, 'img_width:', img_width, 'img_height:', img_height)

            #判断视频是否属于同一个宽度和高度
            # if img_height != 240 or img_width != 320:
            #     print(clip_basename)

            clip5s_num = int(frames_num / clip_5s_frames) + 1
            print(frames_num)
            for j in range(clip5s_num):

                start_clip5s_id += 1
                # save_clip5s_name = clip_5s_root + '/' + clip_basename.split('_')[0] + '_' + str(start_clip5s_id).zfill(5) + '.mp4'
                save_clip5s_name = clip_5s_root + '/' + abnormal_name + '_' + str(start_clip5s_id).zfill(5) + '.mp4'
                # print(save_clip5s_name)


                if j + 1 < clip5s_num:

                    start_frame = j * clip_5s_frames
                    end_frame = (j+1) * clip_5s_frames
                    print(start_frame,end_frame)

                else:
                    start_frame = frames_num - clip_5s_frames
                    end_frame = frames_num
                    print(start_frame, end_frame)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                cap = cv2.VideoCapture(clip_name)
                video_writer = cv2.VideoWriter(save_clip5s_name, fourcc, clip_5s_fps, img_size)
                success = True
                frame_count = 0
                while (success):
                    success, frame = cap.read()
                    if frame_count >= start_frame and frame_count < end_frame:
                        video_writer.write(frame)

                    frame_count += 1

                video_writer.release()


        # if i>=1:
        #     break


def read_clip5s():


    clip5s_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut/for3/Fire'

    clip5s_names = sorted(glob(clip5s_root + '/*.mp4'))
    index = 0
    for i,clip5s_name in enumerate(clip5s_names):

        cap = cv2.VideoCapture(clip5s_name)

        frames_fps = int(cap.get(5))  # 得到帧率
        frames_num = int(cap.get(7))  # 得到总帧数
        img_width = int(cap.get(3))  # 图像宽度
        img_height = int(cap.get(4))  # 图像高度
        print('frames_num', frames_num, 'frames_fps:', frames_fps, 'img_width:', img_width, 'img_height:', img_height)
        index += (frames_num/120)+1

        print(clip5s_name,index)
        # if i>5:
        #     break


def change_foldername():
    folder_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/RWF/Fight_test'
    folder_names = sorted(glob(folder_root + '/*'))

    start_id = 1606
    for i, folder_name in enumerate(folder_names):
        # print(folder_name)
        # folder_basename = os.path.basename(folder_name).split('_')[0]
        change_folder_name = 'Fighting'
        new_folder_name = folder_root + '/' + change_folder_name + '_' +str(start_id).zfill(5)
        print(new_folder_name)

        os.rename(folder_name,new_folder_name)
        start_id +=1
        # if i==0:
        #     break

def visdrone2normal():

    data_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/for3/visdrone'
    save_root = '/media/hp/085d9636-47e0-4970-96d1-5bf4b5c53b4d/u421/user/cc/dataset/UCF_cut5s_frames/for4/Visdrone'

    three_files = sorted(glob(data_root +'/*'))
    start_clip5s_id = 3926
    for i,three_file in enumerate(three_files):
        videos = sorted(glob(three_file + '/*'))
        for j,video in enumerate(videos):

            frames = sorted(glob(video+'/*.jpg'))
            if(len(frames)<120):
                cut_frames_num = len(frames)
            else:
                cut_frames_num = int(len(frames)/120) * 120

            frame_id = 1
            save_video_path = ''
            print(video)
            print(len(frames),cut_frames_num)
            for k,frame in enumerate(frames):
                if k == cut_frames_num:
                    break

                if(k%120==0):
                    # print(k)
                    try:
                        start_clip5s_id += 1
                        frame_id = 1
                        save_video_path = save_root + '/Normal_' + str(start_clip5s_id).zfill(5)
                        os.mkdir(save_video_path)

                    except OSError:
                        pass

                img = cv2.imread(frame)
                save_img_path = save_video_path + '/' + str(frame_id).zfill(5) +'.jpg'
                frame_id += 1
                # print(save_img_path)
                cv2.imwrite(save_img_path,img)



        #     if j==0:
        #         break
        # if i==0:
        #     break


if __name__ == "__main__":


    #2、根据标注裁剪长视频
    # cut_video_clip()

    #1、去除标注文档txt的逗号
    # change_ann()

    #改变文件名称
    # change_filename()


    #将裁剪后的短视频处理成5S的视频
    # transform_videotime(abnormal_name)
    # labels = ['StreetRobbery']
    # #
    # for label in labels:
    #     print(label)
    #     transform_videotime(label)

    #读取短片段，查看信息
    # read_clip5s()

    #将normal_video裁剪成5s视频
    # transform_videotime('Normal')

   #修改文件夹名称
   change_foldername()

   #将visdrone序列转换为normal
   # visdrone2normal()





