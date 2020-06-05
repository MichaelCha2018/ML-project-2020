### run DQN

`python main.py`



### hyper-parameters

```shell
A simple DQN

optional arguments:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
  --gamma GAMMA
  --buffersize BUFFERSIZE
  --startepoch STARTEPOCH
  --dqn_freq DQN_FREQ
  --framelen FRAMELEN
  --dqn_updatefreq DQN_UPDATEFREQ
  --lr LR
  --alpha ALPHA
  --eps EPS
  --imgDIM IMGDIM
  --seed SEED
```



### code

```
yejb
├── Buffer.py       # the replay buffer for DQN
├── README.md
├── main.py         # main program
├── models.py       # store DQN models
├── procedure.py    # the train process
├── utils.py        # some utils functions
├── world.py        # parse hyper-parameters and make it global
└── wrapper.py      # pre-process gtym.Env
```

