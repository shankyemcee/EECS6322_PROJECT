# EECS6322_PROJECT

## Steps To Run

### Step 1. Create a new python environment in the current directory using the command:

```
python -m venv .
```

### Step 2. Activate the environment using the following commands on windows:

```
cd scripts
activate
```

and on linux

```
source bin/activate
```

### Step 3. Install the requirements using the following command:

```
pip install -r requirements.txt
```

and install pytorch directly from source

```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 4: Add model checkpoint directory

```
mkdir model_checkpoints
```

and point the config.ini file to this folder:

```
checkpoint_dir = model_checkpoints/
```

### Step 5. Train stage 1 

First stage of model should be run for 10 epochs, batch size 6. Check the config.ini file to verify this:

```
epochs = 10
batch_size = 6
stage = 1
```

Then run the train.py python file to start training using the following command:

```
python train.py --config_file config.ini --section DEFAULT
```


### Step 6. Train stage 2 

Second stage of model should be run for 15 epochs, batch size 3. Change the config.ini file to reflect this:

```
epochs = 15
batch_size = 3
stage = 2
model_stage1_file = C2F_stage1_epoch9.pt
```

Then run the train.py python file to start training again using the following command:

```
python train.py --config_file config.ini --section DEFAULT
```


### Step 7. Test Model 

To test the saved model, point the config.ini file to the final saved model:

```
model_test_file = C2F_stage2_epoch14.pt
```

Then run the test.py python file to run test phase using the following command:

```
python test.py --config_file config.ini --section DEFAULT
```

Generated summaries will be the GPT_C2F_output.json file in the output folder specified by config.ini file:

```
test_output_dir = output/
```



## Steps To Run Stretch Project

Repeat steps 5-7 but replace train.py and test.py with train_stretch.py and test_stretch.py