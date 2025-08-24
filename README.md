# Midas & IHOP<sup>M</sup>: High-Accuracy Inference Attacks Using Minimal Leakage Against SSE

------

### Implementations of Midas & IHOP<sup>M</sup>

This repository contains Python 3.9 implementation of the attacks presented in:

**Boosting Query Recovery against Encrypted Databases: High-Accuracy and Efficient Inference Attacks Using Minimal Leakage** 


### Install Required Packages

Before running, you need to install the necessary packages

```shell
pip3 install -r requirements.txt
```

### Summary of files

#### Project structure

```angular2html
Midas/
├─attacks
│  │  ihop.py
│  │  ihopM.py
│  │  ikk.py
│  │  jigsaw.py
│  │  midas.py
│  │  midas_incremental_optimization.py
│  │  sap.py
│  │  score.py
│
├─Datasets
│      Enron_3000.pkl             # Contains 3000 keywords/queries
│      Lucene_3000.pkl            # Contains 3000 keywords/queries
│      Lucene_6000.pkl            # Contains 6000 keywords/queries
│
├─pic
│  │  ablationTest.py
│  │  auxS1EnronPlot.py
│  │  auxS1LucenePlot.py
│  │  auxS2EnronPlot.py
│  │  auxS2LucenePlot.py
│  │  auxS3EnronPlot.py
│  │  auxS3LucenePlot.py
│  │  clrzS1EnronPlot.py
│  │  clrzS1LucenePlot.py
│  │  CRPlot.py
│  │  ihopMPlot.py
│  │  limitedTimePlot.py
│  │  mS1EnronPlot.py
│  │  mS1EnronTimePlot.py
│  │  mS1LucenePlot.py
│  │  mS1LuceneTimePlot.py
│  │  mS2EnronPlot.py
│  │  mS2EnronTimePlot.py
│  │  mS2LucenePlot.py
│  │  mS2LuceneTimePlot.py
│  │  mS3EnronPlot.py
│  │  mS3EnronTimePlot.py
│  │  nS1EnronPlot.py
│  │  nS1EnronTimePlot.py
│  │  nS2EnronPlot.py
│  │  nS2EnronTimePlot.py
│  │  nS3EnronPlot.py
│  │  nS3EnronTimePlot.py
│  │  PRaPlot.py
│  │  PRbPlot.py
│  │  prioriQueriesPlot.py
│  │  RRPlot.py
│  │
│  └─pictures
│          *.pdf        
└─pic_pkl
│        *.pkl
│
│  auxTest.py
│  ablationLeakageTest.py
│  ablationOptimizeMTest.py
│  ablationOptimizeNTest.py
│  ablationRRTest.py
│  clrzTest.py
│  CRTest.py
│  ihopMTest.py
│  mTest.py
│  mTestLimitedTime.py
│  nTestLimitedTime.py
│  nTest.py
│  PRTest.py
│  RRTest.py
│  nTestLimitedTimeOnLargeKeyword.py
│  utils.py
```


#### About each file

The code uses two datasets (`Enron` or `Lucene`), with the following three scenarios: 

- **S1**: The keyword universe $\mathsf{W}$ consists of $n$ randomly selected keywords from $\mathcal{U}$. The keywords of $m$ ($m ≤ n$) observed queries are uniformly distributed over $\mathsf{W}$.
- **S2**: The keyword universe $\mathsf{W}$ comprises the top $n$ most frequent keywords in $\mathcal{U}$. The keywords of $m ≤ n$ observed queries are uniformly distributed over $\mathsf{W}$.
- **S3**: The selected keyword set $\mathsf{W} \in U$ is identical to S2, but $m ≤ n$ observed queries correspond to the top $m$ most popular keywords in $\mathsf{W}$.

`attacks`: includes the required attack (`Midas`, `Jigsaw`, `IHOP`, `Score`, `IHOP`<sup>M</sup>,  `IKK`, and `SAP`) implementation.

`pic_pkl`: is used to save the running results of the evaluation script to pickle files.

`auxTest.py`: evaluates the impact of the attacker's auxiliary knowledge (the number of non indexed documents, $|\mathsf{D}_{sim}|$) on attacks in different datasets and scenarios, including recovery accuracy and attack time.

`mTest.py`: evaluates the impact of query quantity ($m$) on attacks in different datasets and scenarios, including recovery accuracy and attack time.

`nTest.py`: evaluates the impact of keyword quantity ($n$) on attacks in different datasets and scenarios, including recovery accuracy and attack time.

`clrzTest.py`: evaluates the impact of CLRZ defense on attacks in different datasets and scenarios.

`PRTest.py`: tests the parameter $\rho$ of `PR`.

`RRTest.py`: tests the parameter $\eta$ of `RR`.

`CRTest.py`: tests the parameters $\mu$ and $\gamma$ of `CR`.

`ihopMTest.py`: tests the parameter $\theta$ of `IHOP`<sup>M</sup>.

`./pic/*Plot.py`: uses pickle files from `pic_pkl` to plot figures of various evaluation scripts.


### Attacks parameters (keys and values)
This is a list, for each attack, of its parameters and values of each, in dictionary format: 

- `'ikk'`: {*'initial temperature'*: 200, *'cooling rate'*: 0.999, *'termination temperature'*: $10^{-10}$, *'rejection threshold'*: 1500}, 
- `'score'`: {*'RefSpeed'*: 10},
- `'sap'`: { $'\alpha'$: 0},
- `'ihop'`: {*'pfree'*: 0.25, *'niters'*: 1000},
- `'jigsaw'`: { $'\alpha'$: 1, $'\beta'$: 0.9, *'BaseRec'*: 45, *'ConfRec'* = 35, *'RefSpeed'*: 10},
- `'midas'`: { $'\rho'$: 20, $'\sigma'$: 55, $'\eta'$: 1, $'\gamma'$: 4, $'\mu'$: 25, $'\delta'$: 10},
- `'midas_incremental_optimization'`: { $'\rho'$: 20, $'\sigma'$: 55, $'\eta'$: 1, $'\gamma'$: 4, $'\mu'$: 25, $'\delta'$: 10},
- `'ihop`<sup>M</sup>': { $'\theta'$: 5, *'pfree'*: 0.25, *'niters'*: 500},
- `'clrz'` (defense): {*'tpr'*: 0.999, *'fpr'*: $[0.01, 0.02, 0.05]$}

### Run

#### The number and accuracy of prior queries recovered by PR varies with $\rho$ in Enron (Figure 2)

```shell
python3 ./PRTest.py 	
python3 ./pic/PRaPlot.py       # Draw a figure
python3 ./pic/PRbPlot.py       # Draw a figure
```

#### Effect of different parameters on RR accuracy (Figure 3)

```shell
python3 ./RRTest.py 	
python3 ./pic/RRPlot.py       # Draw a figure
```

#### Effect of different parameters on CR accuracy (Figure 4)

```shell
python3 ./CRTest.py 	
python3 ./pic/CRPlot.py       # Draw a figure
```

#### Effect of different parameters on IHOP$^M$ accuracy (Figure 5)

```shell
python3 ./ihopMTest.py 	
python3 ./pic/ihopMPlot.py       # Draw a figure
```

#### Impact of non-co-occurrence leakage for prior queries in PR (Figure 6)

```shell
python3 ./ablationLeakageTest.py --scenarios S1	   	    # --scenarios: optional 'S1' and 'S2'.
python3 ./pic/ablationLeakagePlot.py       # Draw a figure
```

#### Effect of iterative refinement  (Figure 7)

```shell
python3 ./ablationRRTest.py 	
python3 ./pic/ablationRRPlot.py       # Draw a figure
```

#### Incremental computation  (Figure 8)

```shell
python3 ./ablationOptimizeMTest.py 	
python3 ./ablationOptimizeNTest.py       
```

#### Comparison on the amount of auxiliary files $|D_{sim}|$ (Figure 9)

```shell
python3 ./auxTest.py --dataset Enron --scenarios S1 	    # --dataset: optional 'Enron' or 'Lucene'. --scenarios: optional 'S1', 'S2', and 'S3'.
python3 ./pic/auxS*Plot.py       # Draw a figure 
```

*Each subgraph requires running a script, where the dataset and scenario parameters need to be changed. The script will generate `pkl` binary files corresponding to the parameters, and then run each drawing script to obtain figures. `./pic/auxS*Plot.py` represents the combination of all scenes and datasets, such as `auxS1EnronPlot.Py`, `auxS2LucenePlot.py`, etc.*

#### Comparison on keyword space $n$ (Figure 10)

```shell
python3 ./nTest.py --dataset Enron --scenarios S1 	    # --scenarios: optional 'S1', 'S2', and 'S3'.
python3 ./pic/nS*Plot.py       # Draw a figure
python3 ./pic/nS*TimePlot.py       # Draw a figure
```

*Each subgraph requires running a script, where the dataset and scenario parameters need to be changed. The script will generate `pkl` binary files corresponding to the parameters, and then run each drawing script to obtain figures. `./pic/nS*Plot.py` represents the combination of all scenes and datasets, such as `nS1EnronPlot.Py`,  `nS1EnronTimePlot.py`,`nS2LucenePlot.py`, etc.*

**Comparison of the under similar runtime (Appendix)**  

```shell
python3 ./nTestLimitedTime.py -s S1  	       # or -s S2
python3 ./pic/nTestLimitedTimePlot.py       # Draw a figure 
```

#### Comparison under similar runtime on large keyword universes (Figure 11)

```shell
python3 ./nTestLimitedTimeOnLargeKeyword.py -s S1  	       # or -s S2
python3 ./pic/limitedTimeNLargeLucenePlot.py       # Draw a figure 
```

#### Comparison on the number of observed queries $m$ (Figure 12)

```shell
python3 ./mTest.py --dataset Enron --scenarios S1 	    # --dataset: optional 'Enron' or 'Lucene'. --scenarios: optional 'S1', 'S2', and 'S3'.
python3 ./pic/mS*Plot.py       # Draw a figure
python3 ./pic/mS*TimePlot.py       # Draw a figure
```

*Each subgraph requires running a script, where the dataset and scenario parameters need to be changed. The script will generate `pkl` binary files corresponding to the parameters, and then run each drawing script to obtain figures. Note that there is no `--dataset Lucene --scenario S3`.*

#### Comparison under similar runtime on $m$ (Figure 13)

```shell
python3 ./mTestLimitedTime.py -s S1  	       # or -s S2
python3 ./pic/mTestLimitedTimePlot.py       # Draw a figure 
```

#### Comparison against defense (Figure 14 and Table 4/5)

```shell
python3 ./clrzTest.py --dataset Enron --scenarios S1 	    # --dataset: optional 'Enron' or 'Lucene'. --scenarios: optional 'S1', 'S2', and 'S3'.
python3 ./pic/clrzS1*.py       # Draw a figure
```

*Here is the parameter list:*

```shell
--dataset Enron --scenarios S1
--dataset Lucene --scenarios S1
--dataset Lucene --scenarios S2
--dataset Enron --scenarios S3
```



### Datasets

The download links for the datasets are as follows:

- Enron: https://www.cs.cmu.edu/~enron/

- Lucene: http://mail-archives.apache.org/mod_mbox/lucene-java-user

  - ```shell
    # Apache Lucene Dataset
    mkdir apache_ml
    cd apache_ml
    for y in {2002..2022}; do
        for m in {01..12}; do
            wget "http://mail-archives.apache.org/mod_mbox/lucene-java-user/${y}${m}.mbox"
        done
    done
    ```

    Type the above code in the terminal to download the required file.



We have processed these datasets (tokenization, filtering, etc.), and the processed pickle files are located in the `Datasets` directory. 

- `Enron_3000.pkl`: is the statistical result of the `Enron` dataset. 
- `Lucene_3000.pkl` or `Lucene_6000.pkl` : is the statistical result of the `Lucene` dataset. 

They are a list of length `4`. The first two values are the file lists corresponding to the statistical keywords/queries. For example, `'dog': [1,2,3]` indicates that the keyword/query appears in the file `[1,2,3]`. The last two values are non overlapping non-indexed documents and indexed document lists, respectively.

### Links
`SAP`: [simon-oya/USENIX21-sap-code: Python code that we used in our USENIX 2021 paper to evaluate query recovery attacks](https://github.com/simon-oya/USENIX21-sap-code)

`Score`: [MarcT0K/Refined-score-atk-SSE: Code to simulate Score and Refined Score attacks on Searchable Symmetric Encryption (SSE)](https://github.com/MarcT0K/Refined-score-atk-SSE)

`IHOP`: [simon-oya/USENIX22-ihop-code: Python code that we used in our USENIX 2022 paper to evaluate query recovery attacks](https://github.com/simon-oya/USENIX22-ihop-code)

`Jigsaw`: [JigsawAttack/JigsawAttack](https://github.com/JigsawAttack/JigsawAttack)


