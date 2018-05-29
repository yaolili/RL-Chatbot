# coding=utf-8
"""
徐瑞健 2018/3/29
2018/5/14 升级成可以循环查询最后自动使用gpu运行程序的版本
2018/5/15 进行本地化，拒绝107的3号卡
2018/5/20 107识别升级，准确识别到107的完整ip，而不是只检查尾数。现在该程序ICST内网通用。
"""

import os
import tensorflow as tf
import socket

MAX_MEMORY_USED = 2000
MAX_UTIL_USED = 20

# localization 拒绝107的3号卡，除了这相关的代码以外其他代码应该都通用
ip_str = socket.gethostbyname(socket.gethostname())
if ip_str[-1] == '172.31.32.107':
    reject_3 = True
else:
    reject_3 = False


def get_gpu_index():
    lines = os.popen('nvidia-smi --query-gpu=memory.used,utilization.gpu'
                     ' --format=csv,noheader').readlines()

    for gpu_id, status in enumerate(lines):
        status = status.split(', ')
        used_memory = int(status[0].split()[0])
        used_util = int(status[1].split()[0])

        # localization 拒绝107的3号卡，除了这相关的代码以外其他代码应该都通用
        if reject_3 and gpu_id == 3:
            continue

        if used_memory < MAX_MEMORY_USED and used_util < MAX_UTIL_USED:
            return gpu_id, used_memory, used_util

    # print('no available gpu, which one do you want to use?')
    # for gpu_id, line in enumerate(lines):
    #     print(gpu_id.__str__() + ' -> ' + line.strip())
    # return int(input('(or ctrl+c) >'))
    return None


print('======= GPU GETTING =======')
gpu_message = None
while gpu_message is None:
    gpu_message = get_gpu_index()

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_message[0].__str__()
SESS = tf.InteractiveSession()
print(('[ATTENTION]\n decided to use gpu {} \
       \n\tused memory: {}\n\tused util: {}')
      .format(gpu_message[0], gpu_message[1], gpu_message[2]))
print('======= GPU GOTTEN =======')
