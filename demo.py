import pandas as pd
import manual_tree as mt
from sklearn.datasets import load_breast_cancer

# 1.prepare data
cancer = load_breast_cancer()
df1 = pd.DataFrame(cancer['data'], columns=['v'+str(i) for i in range(30)])
df2 = pd.DataFrame(cancer['target'], columns=['Y'])
train = pd.concat([df1, df2], axis=1)
valid = pd.concat([df1, df2], axis=1)

# 2.super parameter
save_path = '/Users/lintaocheng/Desktop/dc_20210531/clt/05_test/save'
model_type = 'PD'      
y_label = 'Y'          
criterion = 'gini'      # decision tree bifurcation criteria
min_samples_leaf = 0.1  # minimum sample size of leaf node
logger=''               # log information

# 3. demo create tree
manaul_obj = mt.Manaul(train, valid, y_label, save_path, model_type, criterion, min_samples_leaf, logger)

# step 1
manaul_obj.get_pool_node_id(0)  # divide the data into two pools 1 and 2 according to the feature V20
manaul_obj.calculate_feature_split(variable_selected='v20', split_value=14)
manaul_obj.save_step_split()

# step 2
manaul_obj.get_pool_node_id(1)  # split pool 1 according to feature v27
manaul_obj.calculate_feature_split(variable_selected='v27', split_value=0.099)
manaul_obj.save_step_split()

#step 3
manaul_obj.get_pool_node_id(2) # split pool 2 according to feature v6
manaul_obj.calculate_feature_split(variable_selected='v6', split_value=0.06)
manaul_obj.save_step_split()



