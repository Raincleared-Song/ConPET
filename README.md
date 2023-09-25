# ConPET
Source code for *ConPET: Continual Parameter-Efficient Tuning for Large Language Models*.

### Prepare the Datasets

Download the data from Google Drive [link](https://drive.google.com/drive/folders/1EmU1ljZe145lhITeCXP1hhxkqCilXf8M?usp=sharing). Put the files in the sub-folder with name `data`. You should get the following structure:

```
ConPET/
|-- data
    |-- fewnerd
    |-- fewrel
```

Note that the other 4 datasets (OntoNotes, BBN, TACRED, ACE 2005) are not available due to the license issue.

To prepare for CRL, run the following commands in advance:

```bash
ln -s ../data CRL/data
ln -s ../scripts CRL/scripts
```

### Prepare the Environment

You can get all the information of our Conda experimental environment in `env.yaml`.

### Meaning of Arguments

The following arguments will be frequently mentioned in commands:

- `<dataset_name>`: name of the dataset, chosen from "fewnerd", "ontonotes", "bbn", "fewrel", "tacred", and "ace"
- `<save_suffix>`: a string that determines the suffix of your saving path of checkpoints and results
- `<seed>`: an integer representing the random seed, we use 100, 200, and 300 in our repeated experiments
- `<device>`: a string representing the device, such as "cuda:0"
- `<infer_device>`: a string representing a device for inference, generally different from `<device>`

### Results of Baselines

For **EMR**, run the following command:

```bash
python3 train_cycle.py --dataset_name <dataset_name> --use_selector 0 --batch_limit_policy 0 \
  --cycle_suffix <save_suffix> --method_type linear --continual_method emr \
  --seed <seed> --start 1 --device <device> --clear
```

For **EA-EMR**, run the following command:

```bash
python3 train_eaemr.py --dataset_name <dataset_name> --start 1 --cycle_suffix <save_suffix> \
  --batch_limit_policy 0 --seed <seed> --start 1 --device <device> --clear
```

For **CRL**, run the following commands:

```bash
cd CRL
python run_continual.py --device <device> --dataset <dataset_name> \
  --log_path <save_suffix> --sample_k 500 --seed <seed>
cd ..
```

For all the above three methods, you should simultaneously run a parallel process of the following command to select memorized samples:

```bash
python3 kmeans_client.py --dataset_name <dataset_name> --method_type linear \
  --cycle_suffix <save_suffix> --local --start 1
```

### Results of Static ConPET

#### EMR* (Static ConPET + EMR)

Run the following command:

```bash
python3 train_cycle.py --dataset_name <dataset_name> --use_selector 0 --batch_limit_policy 2 \
  --cycle_suffix <save_suffix> --method_type linear --continual_method emr \
  --seed <seed> --start 1 --device <device> --clear
```

#### EA-EMR* (Static ConPET + EA-EMR)

Run the following command:

```bash
python3 train_eaemr.py --dataset_name <dataset_name> --start 1 --cycle_suffix <save_suffix> \
  --batch_limit_policy 2 --seed <seed> --device <device> --clear
```

#### CRL* (Static ConPET + CRL)

Run the following commands:

```bash
cd CRL
python run_continual.py --device <device> --dataset <dataset_name> \
   --log_path <save_suffix> --sample_k 500 --seed <seed> --dynamic_sampling
cd ..
```

### Results of Dynamic ConPET

Run the following command:

```bash
python3 train_cycle.py --dataset_name <dataset_name> --use_selector 2 \
  --teacher_forcing --batch_limit_policy 2 \
  --cycle_suffix <save_suffix> --method_type linear --continual_method our \
  --seed <seed> --start 1 --device <device> --infer_device <infer_device>
```

### Results of Limitless

Run the following command:

```bash
python3 train_cycle.py --dataset_name <dataset_name> --use_selector 2 \
  --teacher_forcing --batch_limit_policy 0 \
  --cycle_suffix <save_suffix> --method_type linear --continual_method our \
  --seed <seed> --start 1 --device <device> --infer_device <infer_device>
```
