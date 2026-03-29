# My First Machine Learning Project
### **Brief Introduction**
- I created this repository to share my journey and approach to learning machine learning
and deep learning. 
- It reflects my growing interest in these fields and the steps I've been taking to deepen my knowledge. 
- I'm always open to connecting with others who are exploring similar topics or pursuing opportunities to apply my skills.

### **Self Teaching (Autodidact)**
- Learned foundational concepts in machine and deep learning
- Implemented initial Python programmes
- Explored essential machine learning methods including clustering, cataloging, and anomaly detection
- Studied the neural network, including shallow networks, CNNs and RNNs

ref: [SelfTeaching](SelfTeaching/)

### **Complete Coursera Curse Deep Learning Specialization**
- Neural Networks and Deep Learning
- Improving Deep Neural Networks: hyperparameter Tuning, Regularization and Optimization
- Structuring Machine Learning Projects
- Convolutional Neural Networks
- Sequence Models
![Coursera Deep Learning Specilizaiton](Coursera_Deep_Learning_Zertification.png)

### **Neural Network from Scratch (NumPy)**
- Implemented a fully functional neural network from scratch using NumPy, 
- Learned core principles of forward and backward propagation.
- Understood loss function and gradient computation
- Implemented optimization techniques: SGD and Adam
- Compared with the same network implemented in **PyTorch Lightning**, ref [Notebook](NNFromScratch/np_nn_vs_lightning_nn.ipynb).

### **Convolutional Neural Network (PyTorch Lightning)**
- Built a CNN using modern techniques for image classification
  - Studied CNN architecture with residual connection and batch normalization
  - Explored training strategies, including learning rate scheduling
- Trained and evaluated CNN on CIFAR-10, ref [Notebook](Image%20Classification%20Training%20Pipeline/EvaluationNotebook.ipynb).  
- Addressed overfitting/high variance issues
  - Initial accuracy (Train / Eval / Test): **0.99, 0.8852, 0.8827**
  - Solved by
    - Reducing number of dense layers and units
    - Applying augmentation (RandomRotation, ColorJitter)
    - Using regularization techniques, such as dropout  
    - Implementing early stopping
  - Final accuracy (Train / Eval / Test): **0.91, 0.89, 0.88**
- Model evaluation and performance comparison with standard architectures
  - Compare my CNN with ResNet-18 trained on the same dataset
  - ResNet-18 accuracy (Train / Eval / Test): **0.985, 0.927, 0.919**

### **Transfer Learning**
- Initialized pretrained ResNet18 and ResNet50 models (ImageNet weights)
  - Replaced the final fully connected layer to match CIFAR-10 (10 classes)
  - Frozen all backbone parameters and trained only the classifier head
- Adapted the CIFAR-10 DataModule
  - Resized input images from 32×32 to 224×224 to match pretrained model requirements
  - Applied normalization and data augmentation
- Conducted experiments with frozen backbones
  - Compared ResNet18 vs ResNet50 under identical training settings
  - ResNet50 achieved slightly higher validation accuracy (~82–83%), but introduced significantly higher computational cost due to deeper architecture and larger input resolution
- Performed partial fine-tuning
  - Unfroze layer4 of ResNet18 while keeping earlier layers frozen
  - Achieved substantial performance improvement, reaching ~93.8% validation accuracy
- Learned
  - Frozen pretrained features provide a strong baseline but quickly saturate
  - Partial fine-tuning significantly improves generalization
  - ResNet18 offers the best trade-off between accuracy and efficiency for CIFAR-10
  - Input resolution mismatch (32×32 → 224×224) is the main source of computational overhead

ref: [Transfer Learning](TransferLearning/)

### **Transformer From Scratch**
- Implemented a minimal Transformer decoder from scratch using PyTorch Lightning.
  - 4-layer Transformer decoder
  - Causal masking for autoregressive training
  - Multi-head attention mechanisms
  - Trained on Tiny Shakespeare dataset to generate Shakespeare-style text
- Learned
  - Transformer internals: attention, masking, embeddings
  - Sequence generation with a custom decoder
  
ref: [TransformerFromScratch](TransformerFromScratch/)

### **Production-Ready RAG System**
- Built a RAG system for explaining vehicle lateral dynamics
  - Collected and curated technical documents
  - Split documents into semantically meaningful chunks (~200 words)
  - Used a pretrained embedding model (multilingual-e5-base) to encode chunks
  - Generated a vector database using ChromaDB for fast similarity search
  - Implemented similarity search for query handling
  - Integrated LLaMA3 for answer generation
  - Built FastAPI backend and simple frontend (index.html)
- Learned
  - Chunk size and embedding quality strongly influence retrieval accuracy
  - LLaMA3 can generate clear explanations, but requires careful prompt engineering for technical topics

ref: [RAG](RAG/)