import numpy as np
import re
import matplotlib.pyplot as plt


def depth2xyz(depth_map,depth_cam_matrix,flatten=False,depth_scale=180):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz={}
    xyz['x']=x
    xyz['y']=y
    xyz['z']=z
    # xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
    # #xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz
 
if __name__ == '__main__': 
    # 随便生成一个 分辨率为(1280, 720)的深度图, 注意深度图shape为(1280, 720)即深度图为单通道, 维度为2
    #而不是类似于shape为(1280, 720, 3)维度为3的这种
    pfm_file=open('pfm/00000001.pfm','rb')
    header = pfm_file.readline().decode().rstrip()
    channel = 3 if header == 'PF' else 1
 
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")
 
    scale = float(pfm_file.readline().decode().strip())
    if scale < 0:
        endian = '<'    #little endlian
        scale = -scale
    else:
        endian = '>'    #big endlian
 
    disparity = np.fromfile(pfm_file, endian + 'f')
 
    # print(disparity.shape)
    img = np.reshape(disparity, newshape=(height, width))

    depth_map = np.flipud(img)
 
    depth_cam_matrix = np.array([[2892.33, 0,  823.204],
                                 [0,   2883.18,619.069],
                                 [0,   0,    1]])
    xyz=depth2xyz(depth_map, depth_cam_matrix)
    fig = plt.figure(figsize=(12, 10),dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    X=xyz['x']
    Y=xyz['y']
    Z=xyz['z']
    ax.scatter(X, Y, Z,s= 10,edgecolor="black",marker=".")
    plt.show()

