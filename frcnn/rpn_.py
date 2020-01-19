"""
一般
two stage rpn会筛选部分前景（不分类别）和背景的proposal，把经过rpn reg 的 proposal抠出来进行roi_pooling (align) 然后再得到最后的reg 和 cls
<<<<<<< HEAD
one stage rpn直接就是将anchor当作proposal 直接进行reg cls (所以一般会严重的正负样本不均衡), 参考anchor_.py即可
=======
one stage rpn直接就是将anchor当作proposal 直接进行reg cls (所以一般会严重的正负样本不均衡)
>>>>>>> origin/master
"""
