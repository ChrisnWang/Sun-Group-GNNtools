import argparse
import os

base_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='hyper parameters for gcn.')

parser.add_argument('--edge_path', type=str, default= base_dir + '/data/temp/DP_edges.csv', help='graph_edge.')

parser.add_argument('--attribute_path', type=str, default= base_dir + '/data/temp/DP_attributes.csv', help='attributes.')

parser.add_argument('--logs_dir', type=str, default= base_dir + '/logs/', help='logs_dir.')

parser.add_argument('--checkpoints_dir', type=str, default= base_dir + '/checkpoints/', help='checkpoints_dir.')

parser.add_argument('--seed', type=int, default=2020, help='random_seed.')

parser.add_argument('--epoch_num', type=int, default=10, help='train_epoch_num.')

parser.add_argument('--X_shape', type=int, default=154, help='weight_metric_shape.')

parser.add_argument('--train_start', type=int, default=0, help='train_set_partition_start_index.')

parser.add_argument('--test_start', type=int, default=100, help='test_set_partition_start_index_(train_end).')

parser.add_argument('--hidden_dim_1', type=int, default=4, help='hidden_layer_1_dim_in_GCN.')

parser.add_argument('--hidden_dim_2', type=int, default=2, help='hidden_layer_2_dim_in_GCN.')

parser.add_argument('--lr', type=float, default=0.01, help='learning_rate.')

parser.add_argument('--train_rate', type=float, default=0.8, help='learning_rate.')

args = parser.parse_args()

