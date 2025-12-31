import h5py

# 读取第一个数据
f = h5py.File("data_act/episode_0.hdf5", "r")

print("Keys:", list(f.keys()))
print("Action shape:", f['action'].shape)
print("Qpos shape:", f['observations/qpos'].shape)
print("Image Top shape:", f['observations/images/top'].shape)

f.close()