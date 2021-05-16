# DPLAN-Implementation
This is an implementation of the anomaly detecion algorithm proposed in this paper: ["Deep Reinforcement Learning for Unknown Anomaly Detection"](https://arxiv.org/pdf/2009.06847.pdf). Please let me know if there are any bugs in my code. Thank you:)
### Environment
* cuda==10.2
* cudnn==7.6
* python==3.8.8
* tensorflow==2.2.0
* tensorflow-gpu==2.2.0
* keras-rl2==1.0.5
* pandas==1.2.4
* gym==0.18.0
* scikit-learn==0.24.2
### Dataset
* [CoverType](https://archive.ics.uci.edu/ml/datasets/covertype)
* [HumanActivityRecognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
* [UNSW-NB15](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys)
* [ThyroidDisease](https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease)
### Loading Dataset
* Datasets are preprocessed in the same way that described in the original paper.
* One raw dataset will generate a set of unknown anomaly detection dataset, according to different known anomaly classes in the training dataset.
  * For example, UNSW-NB15:
  
<!DOCTYPE html>
<html lang="en">
   <head>
	 <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/8.0.0/mermaid.min.js"></script>
    </head>
	 
<body>
 <pre><code class="language-mermaid">graph LR
start[UNSW-NB15]-->train[Train Dataset]
start-->test[Test Dataset]
train--Know Analysis-->Analysis[Analysis Unknown Dataset]
train--Know Backdoor-->Backdoor[Backdoor Unknown Dataset]
train--Know DoS-->DoS[DoS Unknown Dataset]
train--Know Other-->...

</code></pre>

<div class="mermaid">graph LR
start[data_path]-->ds1[UNSW-NB15]
start-->ds2[CoverType]
start-->ds3[ThyroidDisease]
start-->ds4[...]
ds1-->Analysis
ds1-->Backdoor
ds1-->DoS
ds1-->Exploits
ds1-->...
ds1-->t1[test dataset]
ds2-->cottonwood
ds2-->douglas-fir
ds2-->t2[test dataset]
ds3-->hypothyroid
ds3-->subnormal
ds3-->t3[test dataset]

</div>
	
</body>
<script>
var config = {
    startOnLoad:true,
    theme: 'forest',
    flowchart:{
            useMaxWidth:false,
            htmlLabels:true
        }
};
mermaid.initialize(config);
window.mermaid.init(undefined, document.querySelectorAll('.language-mermaid'));
</script>

</html>
  * Accordingly, the paths of datasets are set in the following way to be loaded:
    
### Experiment
* Set hyperparameters listed in the file `main.py`.
* Run `python main.py`.