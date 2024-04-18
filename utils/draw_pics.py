import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec 
import cv2

DRAW_LINE = [
    (0, 1, 2),
    (1, 4, 2),
    (4, 7, 2),
    (7, 10, 2),
    (13, 9, 2),
    (13, 16, 2),
    (16, 18, 2),
    (18, 20, 2),
    (20, 22, 2),
    (0, 2, 0),
    (2, 5, 0),
    (5, 8, 0),
    (8, 11, 0),
    (14, 9, 0),
    (14, 17, 0),
    (17, 19, 0),
    (19, 21, 0),
    (21, 23, 0),
    (15, 12, 1),
    (12, 9, 1),
    (6, 3, 1),
    (6, 9, 1),
    (0, 3, 1)
]
I, J, LR = [], [], []
for i in range(len(DRAW_LINE)):
    I.append(DRAW_LINE[i][0])
    J.append(DRAW_LINE[i][1])
    LR.append(DRAW_LINE[i][2])

def draw_pic_single(fig, mydatas, mydatas_shift=None):
    # gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4]) 
    # ax_img = fig.add_subplot(gs[0])
    # ax_img.get_xaxis().set_visible(False)
    # ax_img.get_yaxis().set_visible(False)
    # ax_img.set_axis_off()
    # ax_img.set_title('Input')
    # ax_img.imshow(img, aspect='equal')

    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1500, 1500])
    ax.set_ylim3d([-1500, 1500])
    ax.set_zlim3d([-1500, 1500])
    
    if mydatas:
        for id, mydata in enumerate(mydatas):
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            mydata = mydata[:, [0, 2, 1]]
            # mydata = mydata[:, [1, 0, 2]]
            mydata[..., 1] = -mydata[..., 1]
            # mydata[..., 2] = -mydata[..., 2]

            # Make connection matrix
            for i in np.arange(len(I)):
                x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
                # ax.plot(x, y, z, lw=2, color='#FF0066' if LR[i] == 0 else '#FF0099' if LR[i] == 2 else '#FF00FF')
                ax.plot(x, y, z, lw=3, color='#B4B4B4' if LR[i] == 0 else '#FA2828' if LR[i] == 2 else '#F57D7D')

    if mydatas_shift:
        for id, mydata in enumerate(mydatas_shift):
            mydata = mydata[:, [0, 2, 1]]
            mydata[..., 1] = -mydata[..., 1]

            for i in np.arange(len(I)):
                x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
                ax.plot(x, y, z, lw=1.5, color='#00BFFF' if LR[i] == 0 else '#00BFFF' if LR[i] == 2 else '#00BFFF')
                # ax.plot(x, y, z, lw=2, color='#EED5B7' if LR[i] == 0 else '#EE82EE' if LR[i] == 2 else '#FFC0CB')

    # # set grid invisible
    # ax.grid(None)
    
    # # set X、Y、Z background color white
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    # # set axis invisible
    # ax.axis('off')

    plt.tight_layout()
    # plt.show()
    # plt.pause(0.1)
    # plt.savefig('coup1.png', bbox_inches='tight', pad_inches=0.2, dpi=300)
    
    fig.canvas.draw()
    fig_str = fig.canvas.tostring_rgb()
    image_array = np.frombuffer(fig_str, dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()
    image_array = image_array.reshape(nrows, ncols, 3)

    plt.clf()  #清除图像
    return image_array

def overlay_mask(background_img, prospect_img, img_over_x, img_over_y):
    """
    background_img: bgr背景图
    prospect_img: bgra透明前景图
    """
    back_r, back_c, _ = background_img.shape # 背景图像行数、列数
    if img_over_x > back_c or img_over_x < 0 or img_over_y > back_r or img_over_y < 0:
        print("前景图不在背景图范围内")
        return background_img
    pro_r, pro_c, _ = prospect_img.shape # 前景图像行数、列数
    if img_over_x + pro_c > back_c: # 如果水平方向展示不全
        pro_c = back_c - img_over_x # 截取前景图的列数
        prospect_img = prospect_img[:, 0:pro_c, :] # 截取前景图
    if img_over_y + pro_r > back_r: # 如果垂直方向展示不全
        pro_r = back_r - img_over_y # 截取前景图的行数
        prospect_img = prospect_img[0:pro_r, :, :] # 截取前景图

    # prospect_img = cv2.cvtColor(prospect_img, cv2.COLOR_BGR2BGRA) # 前景图转为4通道图像
    prospect_tmp = np.zeros((back_r, back_c, 4), np.uint8) # 与背景图像等大的临时前景图层

    # 前景图像放到前景图层里
    prospect_tmp[img_over_y:img_over_y + pro_r, img_over_x: img_over_x + pro_c, :] = prospect_img

    _, binary = cv2.threshold(prospect_img, 254, 255, cv2.THRESH_BINARY) # 前景图阈值处理
    prospect_mask = np.zeros((pro_r, pro_c, 1), np.uint8) # 单通道前景图像掩模
    prospect_mask[:, :, 0] = binary[:, :, 3] # 不透明像素的值作为掩模的值

    mask = np.zeros((back_r, back_c, 1), np.uint8)
    mask[img_over_y:img_over_y + prospect_mask.shape[0],
    img_over_x: img_over_x + prospect_mask.shape[1]] = prospect_mask

    mask_not = cv2.bitwise_not(mask)

    prospect_tmp = cv2.bitwise_and(prospect_tmp, prospect_tmp, mask=mask)
    background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    prospect_tmp = cv2.cvtColor(prospect_tmp, cv2.COLOR_BGRA2BGR) # 前景图层转为三通道图像
    return prospect_tmp + background_img # 前景图层与背景图像相加合并