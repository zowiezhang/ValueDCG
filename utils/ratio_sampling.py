import random
import numpy as np 

def ratio_sampling(ls1, ls2, len1, len2, num):

    r1, r2 = np.array([len1, len2]) / (len1 + len2)

    draw_from_list1 = int(num * r1)
    draw_from_list2 = num - draw_from_list1

    drawn_from_list1 = random.sample(ls1, draw_from_list1)
    drawn_from_list2 = random.sample(ls2, draw_from_list2)

    return drawn_from_list1 + drawn_from_list2

def three_ratio_sampling(ls1, ls2, ls3, len1, len2, len3, num):

    r1, r2, r3 = np.array([len1, len2, len3]) / (len1 + len2 + len3)

    draw_from_list1 = int(num * r1)
    draw_from_list2 = int(num * r2)
    draw_from_list3 = num - draw_from_list1 - draw_from_list2

    drawn_from_list1 = random.sample(ls1, draw_from_list1)
    drawn_from_list2 = random.sample(ls2, draw_from_list2)
    drawn_from_list3 = random.sample(ls3, draw_from_list3)

    return drawn_from_list1 + drawn_from_list2 + drawn_from_list3
