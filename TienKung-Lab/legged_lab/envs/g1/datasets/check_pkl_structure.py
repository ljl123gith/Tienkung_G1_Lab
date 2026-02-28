import pickle
# 你的文件路径
path = "/home/kogm/G1DWAQ_Lab/TienKung-Lab/legged_lab/envs/g1/datasets/data/walk/B1_-_stand_to_walk_stageii.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)

print("Content of link_body_list:")
print(data.get('link_body_list', 'Not found'))