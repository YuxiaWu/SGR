import argparse

def parameter_parser():
    # sgcn
    parser = argparse.ArgumentParser(description="Run SGCN.")
    parser.add_argument("--edge-path", nargs="?", default="/storage_fast/yxwu/data/mmdial_signed_new.csv", help="Edge list csv.")
    parser.add_argument("--pretrain-features-path", nargs="?", default="/storage_fast/yxwu/update_SGR/SGCN/output/embedding/embeddings.csv", 
                        help="the pretrained features for SGR.")
    parser.add_argument("--log-path", nargs="?", default="./logs/", help="Log json.")
    parser.add_argument("--epochs", type=int, default=100,
	                    help="Number of training epochs for SGCN. Default is 100.")
    parser.add_argument("--reduction-iterations", type=int, default=30, help="Number of SVD iterations. Default is 30.")
    parser.add_argument("--reduction-dimensions", type=int, default=64, help="Number of SVD feature extraction dimensions. Default is 64.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sklearn pre-training. Default is 42.")
    parser.add_argument("--lamb", type=float, default=1.0, help="Embedding regularization parameter. Default is 1.0.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test dataset size. Default is 0.2.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for SGCN. Default is 0.01.")
    parser.add_argument("--weight-decay", type=float, default=10**-5, help="Learning rate. Default is 10^-5.")
    parser.add_argument("--layers", nargs="+", type=int, help="Layer dimensions separated by space. E.g. 32 32.")
    parser.add_argument("--spectral-features", dest="spectral_features", action="store_true")
    parser.add_argument("--general-features", dest="spectral_features", action="store_false")
    parser.set_defaults(spectral_features=False)
    parser.set_defaults(layers=[64, 64])
    parser.add_argument("--img_sim_thr", type = float, default=0.7, help="the threshold for image similarity")

    # state graph reasoning
    parser.add_argument("--use-cuda", default=True, help="whether use gpu.")
    parser.add_argument("--gpu", default=5, help="gpu id.")
    parser.add_argument("--batch-size", default=64, help="batch size") 
    parser.add_argument("--num-epoch", type=int, default=1000, help="epochs of SGR")
    parser.add_argument("--data-path", default='/storage_fast/yxwu/data', help="dataset path")
    parser.add_argument("--test_GCN_epoch", default=100, help="epochs of SGR")

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate of SGR. Default is 0.001.")
    parser.add_argument("--lr-adjust", type=bool, default=False, help="whether adjust the learing reate.")
    parser.add_argument("--test-every-steps", type=float, default=50, help="run test every 50 steps.")
    parser.add_argument("--save-every-steps", type=float, default=500, help="save the model every 50 steps.") 
    parser.add_argument("--save-every-epochs", type=float, default=1, help="save the model every 5 epochs.") 
    
    parser.add_argument("--len_items", type=float, default=1768, help="the number of venues.") 
    parser.add_argument("--hidden_dim", default=128, help="the hidden dim of user feature.") 
    parser.add_argument("--with_image", action='store_true', help="whether consider the images in dataset") 

    parser.add_argument("--act_single", action='store_true', help="whether predict single action") 
    parser.add_argument("--mlp", action='store_true', help="whether use mlp for recommendation") 
    parser.add_argument("--pre_train", action='store_true', help="whether use mlp for recommendation") 
    
    parser.add_argument("--fc2_in_dim", default=100, help="the hidden dim of user feature.") 
    parser.add_argument("--fc2_out_dim", default=64, help="the hidden dim of user feature.") 
    parser.add_argument("--test", action='store_true', help="train or test.") 
    parser.add_argument("--load_model", action='store_true', help="whether load model for finetuning or testing.") 
    parser.add_argument("--step", default=5, help="step num for finetuning or testing.") 
    parser.add_argument("--test_GCN", action='store_true', help="whether run sGCN in test.") 
    
    parser.add_argument("--request_refine",action='store_true', help="whether refine the request.") 
    parser.add_argument("--neglink_num", type=float, default=1, help="the number of neg links for each slot.") 

    return parser.parse_args()
