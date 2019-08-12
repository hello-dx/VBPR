# VBPR


A simple version of VBPR.

Two files are needed to run "VBPR.py"  

1. "image_feature.csv" The first column of the csv file is item id, while each line is the corresponding feature of the item. 

     This file is to build a dictionary of item_id and item_feature : self.imageFeatures.

2. "feedback_file.json"  The file stores the feedback data of users & items, which has the structure as follows:

        { "user_id1" : { "item_id_1", "item_id_2", ... },
          "user_id2" : { "item_id_i", "item_id_j", ... },
           ... }
     This file is to build the user-item relation map: self.R.

You need to complete the code in line 113-114 with the path of the above files.

baidu drive link: https://pan.baidu.com/s/1FIsLca0TZW_I4wGmfDwpDw cue：5j3g 

google drive link: https://drive.google.com/open?id=1VVQPltHnf3TxlvnZ2WTagORdiqPDqR-a

--------------------------------------------------------------------------------

# The update edition

An end to end VBPR, which is no need to load all image features at begining.

Three files are needed to run "VBPR_update.py"

1. "user_idx.json" & "item_idx.json". Structures are as follows,

   { "user_0": 0, "user_1": 1, ..., "user_n": n-1 }
   { "item_0": 0, "item_1": 1, ..., "item_m": m-1 }
   
2. Your own rating file. The interaction between users and items are needed.


Step to run "VBPR_update.py"

1. model = VBPR(K=?, K2=?)

2. model.load_training_data()  # filled with "user_idx.json" & "item_idx.json"

3. build data fed in placeholders, and feed their to the network.


