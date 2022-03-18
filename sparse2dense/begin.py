#参数设置
from train import train
from train_with_bucket import train_bucket

class Args():
    def __init__(self):
        self.checkpoint = "sparse2dense.pt"
        self.num_layers = 2  #GRU layers
        self.de_layer = 1
        self.hidden_size = 256   #Hidden state size in GRUs
        self.embedding_size = 256    #Cell embedding size
        self.print = 50    #Print for x iters
        self.dropout = 0.1  #Dropout
        self.learning_rate = 0.001
        self.epochs = 10     #The number of training epochs
        self.save = 500    #Save frequency
        self.criterion_name = "CE"
        self.max_num_line =1000000
        self.bidirectional =True    #True for bidirectional rnn in encoder
        self.max_length = 300       #The maximum length of the target sequence
        self.batch = 128
        self.bucketsize = [(50,50),(100,100),(200,200),(300,300),(400,400),(500,500),(600,600)]
        self.input_cell_size = 2245
        self.output_cell_size = 2245
        self.mode = 1
        self.cuda = True
        self.min_lat = 29.59
        self.max_lat = 29.62
        self.min_lon = 106.47
        self.max_lon = 106.55
        self.width = 8900
        self.height = 3336
        self.patience = 10


args = Args()

if args.mode == 1:
    print('train_with_bucket')
    train_bucket(args)
else:
    print('train')
    train(args)



