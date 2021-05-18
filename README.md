# DPLAN-Implementation
This is an implementation of the anomaly detecion algorithm proposed in this paper: ["Deep Reinforcement Learning for Unknown Anomaly Detection"](https://arxiv.org/pdf/2009.06847.pdf). Please let me know if there are any bugs in my code. Thank you:)
## Environment
* cuda==11.0
* cudnn==8.0
* python==3.8.8
* tensorflow-gpu==2.2.0
* keras-rl2==1.0.5
* pandas==1.2.4
* gym==0.18.0
* scikit-learn==0.24.2
## Dataset
* [CoverType](https://archive.ics.uci.edu/ml/datasets/covertype)
* [HumanActivityRecognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
* [UNSW-NB15](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys)
* [ThyroidDisease](https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease)
## Loading Dataset
* Datasets are preprocessed in the same way that described in the original paper.
* One raw dataset will generate a set of unknown anomaly detection dataset, according to different known anomaly classes in the training dataset.
  * For example, UNSW-NB15:
    
    [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBzdGFydFtVTlNXLU5CMTVdLS0-dHJhaW5bVHJhaW4gRGF0YXNldF1cbiAgICBzdGFydC0tPnRlc3RbVGVzdCBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IEFuYWx5c2lzLS0-QW5hbHlzaXNbQW5hbHlzaXMgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IEJhY2tkb29yLS0-QmFja2Rvb3JbQmFja2Rvb3IgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IERvUy0tPkRvU1tEb1MgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IE90aGVyLS0-Li4uIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBzdGFydFtVTlNXLU5CMTVdLS0-dHJhaW5bVHJhaW4gRGF0YXNldF1cbiAgICBzdGFydC0tPnRlc3RbVGVzdCBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IEFuYWx5c2lzLS0-QW5hbHlzaXNbQW5hbHlzaXMgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IEJhY2tkb29yLS0-QmFja2Rvb3JbQmFja2Rvb3IgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IERvUy0tPkRvU1tEb1MgVW5rbm93biBEYXRhc2V0XVxuICAgIHRyYWluLS1Lbm93IE90aGVyLS0-Li4uIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

  * Accordingly, the paths of datasets to be loaded are set in the following way:
    
[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVEQ7XG4gICAgc3RhcnRbZGF0YV9wYXRoXS0tPmRzMVtVTlNXLU5CMTVdXG4gICAgc3RhcnQtLT5kczJbQ292ZXJUeXBlXVxuICAgIHN0YXJ0LS0-ZHM0Wy4uLl1cbiAgICBkczEtLT5BbmFseXNpc1xuICAgIGRzMS0tPkJhY2tkb29yXG4gICAgZHMxLS0-Li4uXG4gICAgZHMxLS0-dDFbdGVzdCBkYXRhc2V0XVxuICAgIGRzMi0tPmNvdHRvbndvb2RcbiAgICBkczItLT5kb3VnbGFzLWZpclxuICAgIGRzMi0tPnQyW3Rlc3QgZGF0YXNldF0iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVEQ7XG4gICAgc3RhcnRbZGF0YV9wYXRoXS0tPmRzMVtVTlNXLU5CMTVdXG4gICAgc3RhcnQtLT5kczJbQ292ZXJUeXBlXVxuICAgIHN0YXJ0LS0-ZHM0Wy4uLl1cbiAgICBkczEtLT5BbmFseXNpc1xuICAgIGRzMS0tPkJhY2tkb29yXG4gICAgZHMxLS0-Li4uXG4gICAgZHMxLS0-dDFbdGVzdCBkYXRhc2V0XVxuICAgIGRzMi0tPmNvdHRvbndvb2RcbiAgICBkczItLT5kb3VnbGFzLWZpclxuICAgIGRzMi0tPnQyW3Rlc3QgZGF0YXNldF0iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)
    
## Experiment
* Set hyperparameters listed in the file `main.py`.
* Run `python main.py`.