
# next_index = [0406,0407,0408,0409,0410,0411,0412,0413,0414,0415,0416,0417,0418,0419,0420,0421,0422,0423,0424,0425,0425,0426,0427,0428,0429,0430,0431,0432,0433,0434,0435,0436,0437,0438,0439,0440,0441,0442,0443,0444,0445,0446,0447,0448,0449,0450
# 3946,3949,3964,3970,3981,3983,3986,3999,4002,4006,4015,4037,4081,4082,4116,4270, 4271]

# del_index = [0074,0075,0076, 0152, 0481,0615, 0640, 0644, 0985, 0735,0749,0751,0814,1120,1135,1151,1152,1287,1306,1307,1364,1365,1366,1367,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1380,1381,1382,1383,1384,1385,1386,1387,1388,1389,1390,1391,1392,1394,1895,1900,1923,1924,1947,1961,1963,1967,2681,2884,3285,3432,3444,3559]

next_index = ["0406","0407","0408","0409","0410","0411","0412","0413","0414","0415","0416","0417","0418","0419","0420","0421","0422","0423","0424","0425","0425","0426","0427","0428","0429","0430","0431","0432","0433","0434","0435","0436","0437","0438","0439","0440","0441","0442","0443","0444","0445","0446","0447","0448","0449","0450","3946","3949","3964","3970","3981","3983","3986","3999","4002","4006","4015","4037","4081","4082","4116","4270","4271"]

del_index = ["0146","0074","0075","0076","0152","0481","0615","0640","0644","0985","0735", "3954","3981", "3995", "4082","0749","0751","0814","1120","1135","1151","1152","1287","1306","1307","1364","1365","1366","1367","1368","1369","1370","1371","1372","1373","1374","1375","1376","1377","1378","1379","1380","1381","1382","1383","1384","1385","1386","1387","1388","1389","1390","1391","1392","1394","1895","1900","1923","1924","1947","1961","1963","1967","2681","2884","3285","3432","3444","3559"]

offset_1_index = []
import shutil
import os
# source_dir = ""
source_dir = "/data2/datasets/hallo/hallo_data/first_frame"
dis_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset"
# for index in next_index:
#     file_path = os.path.join(source_dir, index + "_first_frame.jpg")
#     new_file_path = os.path.join(dis_dir, index + "_first_frame.jpg")
#     shutil.copy(file_path, new_file_path)

del_dir = "/data2/datasets/hallo/hallo_data/first_frame_error_offset2"
# file_lis = []
# for file in os.listdir(del_dir):
#     file_lis.append(file.replace("_first_frame.jpg", ""))
# print(file_lis)

del_file_lis = ['0442', '0411', '0423', '0937', '0416', '0421', '0413', '0462', '0414', '0429', '0441', '0461', '0444', '0426', '0528', '0430', '0447', '0437', '0443', '0412', '0408', '0450', '4081', '0445', '0427', '1306', '0435', '0418', '3999', '3956', '0438', '0422', '0449', '0425', '0938', '0146', '0433', '0417', '0463', '0436', '0939', '0407', '0406', '0460', '0431', '0446', '0440', '0432', '0410', '0428', '0415', '0448', '0424', '0419', '0409', '0420', '0434', '0940', '0439']

# all time

import os
from moviepy.editor import VideoFileClip

def get_total_duration(folder_path):
    total_duration = 0.0  # 用于累积总时长

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为 .mp4 文件
        if file_name.endswith('.mp4'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 打开视频文件并获取时长
                with VideoFileClip(file_path) as video:
                    total_duration += video.duration
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # 将总秒数转换为小时和分钟
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)

    return hours, minutes

# 使用方法
folder_path = '/data2/datasets/hallo/hallo_data/videos'  # 将路径替换为目标文件夹路径
# hours, minutes = get_total_duration(folder_path)
# print(f"Total duration: {hours} hours {minutes} minutes")

## del index
# del_index = del_index + del_file_lis
# for index in del_index:
#     file_path = os.path.join(folder_path, f"{index}.mp4")
#     if os.path.exists(file_path):
#         os.remove(file_path)

# 1*face_width
offset_1 = []
for file in os.listdir("/data2/datasets/hallo/hallo_data/first_frame_error_offset_1"):
    offset_1.append(file.replace("_first_frame.jpg", ""))
print(sorted(offset_1))
['0454', '0455', '0456', '0457', '0458', '0459', '0464', '0465', '0536', '1541', '1543', '1544', '1589', '1590', '1591', '2538', '2691', '2712', '2754', '2777', '2942', '3217', '3219', '3220', '3221', '3222', '3224', '3225', '3226', '3227', '3229', '3230', '3231', '3232', '3235', '3236', '3237', '3238', '3240', '3755', '3954', '3995', '4074', '4110', '4128', '4129', '4130']
