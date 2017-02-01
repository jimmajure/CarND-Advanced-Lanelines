'''
Created on Jan 21, 2017

@author: jim
'''
import glob
from find_lanes import imread, plot_images_2d, Undistorter,\
    PerspectiveTransformer, gradient_thresh, gradient_dir_thresh,\
    gradient_mag_thresh, mask, plot_images, gradient, gradient_magnitude, scale,\
    rgb2hls, process_image, step_prob, max_window, Timer, find_lane,\
    rgb2hsv, rgb2lab
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imwrite, equalizeHist
from scipy.special.basic import comb


src = np.float32(
    [[100,720],
     [580,450],
     [700,450],
     [1180,720]]
    )
dest = np.float32(
    [[200,720],
     [200,0],
     [1080,0],
     [1080,720]]
    )



def combine1(img, gray):
    grdx = gradient_thresh(gray, "x", 5, (30,100))

    _,_,l,s = rgb2hls(img)
    maskS = mask(s,(170,255))
    maskL = mask(l,(10,255))

    comb = np.zeros_like(maskS)
    comb[((maskS == 1)&(maskL==1)) | (grdx == 1)] = 1
    
#     return [('grdx',grdx),('maskS',maskS),('grdx | maskS',comb)]
    return [('grdx | maskS',comb)]

def combine2(img, gray):
    grdx = gradient_thresh(gray, "x", 5, (20,100))
    grdy = gradient_thresh(gray, "y", 5, (20,100))
    mag = gradient_mag_thresh(gray, ksize=7, mask_range=(30,100))
    dir = gradient_dir_thresh(gray, ksize=9, mask_range=(0.7,1.3))

    hls,h,l,s = rgb2hls(img)
    maskS = mask(s,(170,255))
    maskH = mask(h,(15,100))

    comb = np.zeros_like(maskS)
    comb[((maskS == 1)&(maskH==1)) | (grdx == 1)] = 1
    
    return ('grdx|(maskS & maskH)',comb)

def combine3(img, gray):
    grdx = gradient_thresh(gray, "x", 5, (20,100))
    grdy = gradient_thresh(gray, "y", 5, (20,100))
    mag = gradient_mag_thresh(gray, ksize=5, mask_range=(30,100))
    dir = gradient_dir_thresh(gray, ksize=5, mask_range=(1.0,1.7))

    hls,h,l,s = rgb2hls(img)
    maskS = mask(s,(150,255))
    maskH = mask(h,(15,20))
    maskL = mask(l,(30,40))

    comb = np.zeros_like(maskS)
    comb[((maskH == 1))] = 1
    
    return ('(yellowish&dir)',comb)

def lanes_rsvl(rgb):
    r = rgb[:,:,0]
    req = cv2.equalizeHist(r)
    rmask = mask(req, (252, 255))
    
    hls,h,l,s = rgb2hls(rgb)
    seq = cv2.equalizeHist(s)
    smask = mask(seq, (252, 255))
    
    hsv,h,s,v = rgb2hsv(rgb)
    veq = cv2.equalizeHist(v)
    vmask = mask(veq, (252, 255))
    
    lab,l,a,b = rgb2lab(rgb)
    leq = cv2.equalizeHist(l)
    lmask = mask(leq, (252, 255))
    
    comb = np.zeros_like(r)
    comb[(rmask==1)|(vmask==1)|(lmask==1)] =1

    return comb

def plot_rsvl(rgb, name):
    r = rgb[:,:,0]
    req = cv2.equalizeHist(r)
    rmask = mask(req, (252, 255))
    dsp_images = [(name+'_req',req),('rmask', rmask)]
    
    hls,h,l,s = rgb2hls(rgb)
    seq = cv2.equalizeHist(s)
    smask = mask(seq, (253, 255))
    dsp_images += [(name+'_seq',seq),('smask', smask)]
    
    hsv,h,s,v = rgb2hsv(rgb)
    veq = cv2.equalizeHist(v)
    vmask = mask(veq, (252, 255))
    dsp_images += [(name+'_veq',veq),('vmask', vmask)]
    
    lab,l,a,b = rgb2lab(rgb)
    leq = cv2.equalizeHist(l)
    lmask = mask(leq, (252, 255))
    dsp_images += [(name+'_leq',leq),('lmask', lmask)]
    
    comb = np.zeros_like(r)
    comb[(rmask==1)|(vmask==1)|(lmask==1)] =1
    dsp_images += [('comb',comb)]
    plot_images(dsp_images)
    plt.show()



def plot_colorspaces(rgb,name):
    r = rgb[:,:,0]
    req = equalizeHist(r)
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    dsp_images= [(name+'_rgb',rgb),('r',r),('req',req),('g',g),('b',b)]
    plot_images(dsp_images)
    plt.show()
    
    hls,h,l,s = rgb2hls(rgb)
    seq = cv2.equalizeHist(s)
    dsp_images= [(name+'_hls',hls),('h',h),('l',l),('s',s),('seq',seq)]
    plot_images(dsp_images)
    plt.show()
    
    hsv,h,s,v = rgb2hsv(rgb)
    veq = cv2.equalizeHist(v)
    dsp_images = [(name+'_hsv',hsv),('h',h),('s',s),('v',v),('veq',veq)]
    plot_images(dsp_images)
    plt.show()
    
    lab,l,a,b = rgb2lab(rgb)
    leq = cv2.equalizeHist(l)
    dsp_images= [(name+'_lab',lab),('l',l),('leq',leq),('a',a),('b',b)]
    plot_images(dsp_images)
    plt.show()

def plot_grayscale(rgb, name):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    grdx = scale(gradient(gray_eq, orient='x', ksize=7))
    grdmag = scale(gradient_magnitude(gray_eq, ksize=7))
    
    dsp_images= [(name+'_gray',gray),('gray_eq',gray_eq),('gradient',grdx),('gradient mag',grdmag)]
    plot_images(dsp_images)
    plt.show()

def plot_lab(rgb, name):
    lab,l,a,b = rgb2lab(rgb)
    gradl = scale(gradient(l, orient="x", ksize=11))
#     grdlow = mask(gradl,(0,100))
#     grdhigh = mask(gradl,(135,255))
#     comb = np.zeros_like(gradl)
#     comb[(grdlow==1)|(grdhigh==1)] = 1
    
    maskl = mask(l, (190,255))
    
    dsp_images= [(name+'_l',l),('maskl',maskl),('gradl',gradl)]
    plot_images(dsp_images)
    plt.show()
    

if __name__ == '__main__':
    undistorter = Undistorter('camera_cal/distortion_pickle.p')
    per_trans = PerspectiveTransformer(src, dest, (1280,720))
    image_names = glob.glob('test_images/test*.jpg')
#     image_names = ['test_images/straight_lines1.jpg','test_images/straight_lines2.jpg']
#     image_names = ['test_images/test2.jpg']
 
    for image_name in image_names:
        img = imread(image_name)
        img_und = undistorter.undistort(img)
        
#         plotter = plot_colorspaces
        plotter = plot_rsvl
#         plotter = plot_lab
#         plotter = plot_grayscale
        plotter(img_und, image_name)

        img_lns = lanes_rsvl(img_und)
        img_pt = per_trans.transform(img_lns)
        plot_images([("pic",img_und),("pt lane",img_pt)])
        plt.show()
        
        if False:
            timer = Timer()
            lanes = lanes_rsvl(img_pt)
            print("lanes_rsvl: {}".format(timer.lap()))
            left = find_lane(lanes, 315)
            print("left: {}".format(timer.lap()))
            left_fit = fit_lane(left)
            print("left fit: {}".format(timer.lap()))
            right = find_lane(lanes, 1060)
            print("right: {}".format(timer.lap()))
            right_fit = fit_lane(right)
            print("right fit: {}".format(timer.lap()))
            dsp_images = [(image_name, lanes),('probs',left)]
            plot_images(dsp_images)
       
            ys = np.array(range(720))*1.
            left_fitx = left_fit[0]*ys**2 + left_fit[1]*ys + left_fit[2]
            right_fitx = right_fit[0]*ys**2 + right_fit[1]*ys + right_fit[2]
               
            plt.imshow(lanes, cmap="gray")
            plt.plot(left_fitx, ys, color="red", linewidth=1)
            plt.plot(right_fitx, ys, color="red", linewidth=1)
               
            plt.show()
        
#         gray = cv2.cvtColor(img_pt, cv2.COLOR_RGB2GRAY)
#         grdx = scale(gradient(gray, orient="x", ksize=3))
#         grdx5 = scale(gradient(gray, orient="x", ksize=5))
#         grdx7 = scale(gradient(gray, orient="x", ksize=7))
#         hls,h,l,s = rgb2hls(img_pt)
# 
#         dsp_images = [(image_name.split('/')[-1],img_pt),("hls",hls)]
# 
#         dsp_images += combine1(img_pt, gray)
#         plot_images(dsp_images)
# #         plt.imshow(scale(gradient(gray, orient="x", ksize=9)))
#         plt.show()

#     for image_name in image_names:
#         timer = Timer()
#           
#         img = imread(image_name)
#         img_und = undistorter.undistort(img)
#         img_pt = per_trans.transform(img_und)
#         gray = cv2.cvtColor(img_pt, cv2.COLOR_RGB2GRAY)
#         lanes = combine1(img_pt, gray)[0][1]
#         left = find_lane(lanes, 315)
#         left_fit = fit_lane(left)
#         right = find_lane(lanes, 1060)
#         right_fit = fit_lane(right)
#         print ("time: {}".format(timer.lap()))
#         dsp_images = [(image_name, lanes),('probs',left)]
#         plot_images(dsp_images)
#   
#         ys = np.array(range(720))*1.
#         left_fitx = left_fit[0]*ys**2 + left_fit[1]*ys + left_fit[2]
#         right_fitx = right_fit[0]*ys**2 + right_fit[1]*ys + right_fit[2]
#           
#         plt.imshow(lanes, cmap="gray")
#         plt.plot(left_fitx, ys, color="red", linewidth=1)
#         plt.plot(right_fitx, ys, color="red", linewidth=1)
#           
#         plt.show()
         
