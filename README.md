# SGR

code for "State Graph Reasoning for Multimodal Conversational Recommendation" TMM 2022


# Requirements and Installation

We run the experiment in Pytorch.

The package can be installed by running the following command.

`pip install -r requirements.txt`


# Dataset 

The original dataset we used is： "MMConv: An Environment for Multimodal Conversational Search across Multiple Domains", SIGIR 21

We provide the preprocessed data files which are saved in `./dialogpt/resources.zip` including the following files:

action prediction data:

`train.action_prediction`, `val.action_prediction`, `test.action_prediction`


# Pretrain the signed for the knowledge graph

```
cd ./SGCN/src/

python main.py

```
You will get the embeddings of each node in the graph in `./SGCN/output/embedding/embeddings.csv`

# Action prediction

**The checkpoint of action predication can be downloaded from here** :

link: https://pan.baidu.com/s/1nlMlE8ge9HfV6sqpuhiJDA?pwd=kwqx Code: kwqx

Put it into the fold 'dialogpt/checkpoint/'

**Run `python get_action_predction_results.py`**. You will get the action prediction results `data/act_prediction_result/`

You can also train the action prediction model using the data files in `./dialogpt/resources`: 
```
cd ./dialogpt
python train_ap.py
```

# SGR model training

`bash train.sh`:  train the model by main.py

`bash test.sh`:  run inference 

`bash online.sh`:  run the online conversation by the online environment


# Citation

```
@article{wu2022state,
  title={State graph reasoning for multimodal conversational recommendation},
  author={Wu, Yuxia and Liao, Lizi and Zhang, Gangyi and Lei, Wenqiang and Zhao, Guoshuai and Qian, Xueming and Chua, Tat-Seng},
  journal={IEEE Transactions on Multimedia},
  volume={25},
  pages={3113--3124},
  year={2022},
  publisher={IEEE}
}

```



