
import fenchi_server as server


class Manaul(object):
    def __init__(self, train, valid, y_label, save_path, model_type, criterion, min_samples_leaf, logger):
        model_type = model_type.upper()  
        self.train = train
        self.valid = valid
        self.y_label = y_label
        self.save_path = save_path
        self.model_type = model_type
        self.criterion = criterion                 
        self.min_samples_leaf = min_samples_leaf   
        self.logger = logger                       
        self.node_id = 0
        self.tree_data = {}
        self.sample_tot = self.train.shape[0]      
        self.event_tot = sum(train[y_label]==1) if (model_type=='PD') else None   

    def get_pool_node_id(self, node_id):
        self.node_id = node_id
        
        if self.node_id == 0:
            X = self.train.drop([self.y_label], 1)   
            Y = self.train[self.y_label]             
            self.tree_data[0] = server.init_zero_node_info(self.model_type, Y, self.criterion)  # first node of decision tree
            self.sub_train =self.train
        else:
            if self.node_id not in self.tree_data:
                raise NameError(f'node:{self.node_id} non-existent，ensure that the parent node has been saved ！')
            indicator = server.parse_condition_str(self.tree_data[self.node_id]['node_desc'], self.train)  # meet the sample index corresponding to the split value
            self.sub_train = self.train.loc[indicator == 1]
            X = self.sub_train.drop([self.y_label], 1)
            Y = self.sub_train[self.y_label]

        self.variable_importance  = server.calculate_feature_importance(self.model_type, X, Y, self.criterion)
        return self.variable_importance
    
    def calculate_feature_split(self, variable_selected, split_value):
        self.variable_selected = variable_selected
        self.split_value = split_value
        if self.node_id not in self.tree_data:
            raise NameError(f'node:{self.node_id} non-existent，ensure that the parent node has been saved ！')
        self.left_child, self.right_child = server.calculate_manual_split(
                                                                        self.model_type,     
                                                                        variable_selected,   # feature 
                                                                        split_value,         # feature splitting value
                                                                        self.sub_train,
                                                                        self.y_label,
                                                                        self.criterion,
                                                                        self.node_id,
                                                                        self.tree_data[self.node_id]['node_desc'],  # eg【v22 <= 70】                                                             
                                                                        max(self.tree_data.keys()),  # maximum node label
                                                                        self.sample_tot,   
                                                                        self.event_tot     
                                                                        )
        return None

    def save_step_split(self):
        if self.node_id not in self.tree_data:
            raise NameError(f'node:{self.node_id} non-existent，ensure that the parent node has been saved ！')
       
        self.tree_data[self.node_id]['split_var'] = self.variable_selected
        self.tree_data[self.node_id]['split_value'] = self.split_value
        self.tree_data[self.left_child['node_id']] = self.left_child
        self.tree_data[self.right_child['node_id']] = self.right_child                

        p = server.manual_tree_plot(self.model_type, self.tree_data.values(), self.criterion, self.save_path, 'save_new_tree')

        return p







