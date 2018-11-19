# VBPR


A simple version of VBPR.

Two files are needed to run "VBPR.py"

1. "image_feature.csv" The first column of the csv file is item id, while each line is the corresponding feature of the item. This file is to build a dictionary of item_id and item_feature : self.imageFeatures.

2. "feedback_file.json"  The file stores the feedback data of users & items, which has the structure as follows:

        { "user_id1" : { "item_id_1", "item_id_2", ... },
   
          "user_id2" : { "item_id_i", "item_id_j", ... },

           ... }

     This file is to build the user-item relation map: self.R.

You need to complete the code in line 113-114 with the path of the above files.