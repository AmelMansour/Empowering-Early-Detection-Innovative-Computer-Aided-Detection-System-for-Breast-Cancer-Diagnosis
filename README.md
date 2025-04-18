# Empowering-Early-Detection-Innovative-Computer-Aided-Detection-System-for-Breast-Cancer-Diagnosis

This research presents a novel approach to breast cancer diagnosis using a computer-aided detection (CAD) system that leverages mammographic images for early detection and enhanced diagnostic accuracy. The study highlights the transformative potential of artificial intelligence (AI), particularly deep learning techniques, in early cancer detection, offering a robust tool to improve patient outcomes. The developed system employs convolutional neural networks (CNNs) and incorporates a pre-trained ResNet152 model to extract and classify features from mammographic images, distinguishing between benign and malignant tumors. To further enhance the model’s performance, the CNN model’s hyperparameters are optimized using advanced metaheuristic algorithms, including genetic algorithms, particle swarm optimization, simulated annealing, and tabu search. A key aspect of this work is the utilization of the Optuna framework, an efficient hyperparameter optimization tool, which significantly contributed to achieving remarkable results, with an accuracy of 99.84%, precision of 99.86%, recall of 99.85%, and an F1 score of 99.85%. The experimentswere conducted on two publicly available datasets, namely the Digital Database for Screening Mammography (DDSM) and the INbreast dataset. These datasets provide a diverse set of mammographic images, making them suitable for the comprehensive evaluation of the developed CAD system. The optimization capabilities of Optuna played a crucial role in improving the model’s performance and convergence speed, making it a promising tool for clinical applications.

# Objectives:

* Develop a computer-aided detection (CAD) system to differentiate between benign and malignant breast tumors.

* Enhance the accuracy and efficiency of breast cancer diagnosis.

* Utilize deep learning models, specifically Convolutional Neural Networks (CNN) and ResNet152, for image classification.

* Optimize the model’s hyperparameters using various metaheuristic algorithms:
Genetic Algorithms (GA), Particle Swarm Optimization (PSO), Simulated Annealing (SA), Tabu Search (TS) and Optuna framework

* Evaluate the model’s performance using the following metrics: Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC Curve , AUC

# Methodology:

Data:

DDSM: 13,128 images (PNG, 50μm/pixel).

INbreast: 7,632 images (DICOM, 70μm/pixel).

![image](https://github.com/user-attachments/assets/96d7cd99-45d2-478d-8606-698e2fffee3d)

# Results:
# CNN without optimization algorithms:

Hyperparameters used:

* Learning Rate: 0.0001

* Batch Size: 32

* Number of Epochs: 10

* Optimizer: Adam

* Loss Function: Binary Crossentropy


![image](https://github.com/user-attachments/assets/fc20b30d-9f63-480f-93e3-c3be83ab4917)


# CNN with hyperparameter optimization algorithms:

Algorithms used: Genetic Algorithm (GA), Simulated Annealing (SA), Tabu Search (TS), Particle Swarm Optimization (PSO), Optuna.

![image](https://github.com/user-attachments/assets/8fb30d07-686a-435a-9bfa-3919c42bbe43)


# ResNet152 model without hyperparameter optimization:

![image](https://github.com/user-attachments/assets/e830fbfa-3c25-41fe-9240-4d86245b6ee6)


# ResNet152 model with Optuna:

![image](https://github.com/user-attachments/assets/cc50cfd4-a0fe-4cd4-897a-240a50f2ed7e)


# ResNet152 fine-tuning on INbreast: 

![image](https://github.com/user-attachments/assets/466c55ea-59e0-4e19-a944-60d6b917d65b)


Improvement over baseline ResNet152:

![image](https://github.com/user-attachments/assets/122bba4c-a4e4-493f-a6d1-e9cb3bc2c13d)

# External comparison: comparing our approach against state-of-the-art studies

![image](https://github.com/user-attachments/assets/a900b087-fac2-4a06-87ee-d8aa77a12573)


# Conclusion: 
This thesis presented a deep learning-based system for breast cancer detection using mammographic
images. The proposed methodology leveraged convolutional neural networks
optimized through metaheuristic algorithms, such as genetic algorithm, particle swarm optimization,
simulated annealing, tabu search, and the Optuna framework. The use of these
optimization techniques significantly improved the performance of the CNN model by finetuning
its hyperparameters, leading to a notable increase in classification accuracy. The
optimized model achieved an impressive classification accuracy of 99.84%, along with high
precision (99.86%), recall (99.85%), and F1-Score (99.85%), demonstrating the system’s
potential for reliable breast cancer diagnosis.
The system was evaluated on two publicly available mammographic datasets, DDSM and
INbreast. While the model performed exceptionally well on DDSM, the results on the INbreast
were less favorable, prompting further investigation and fine-tuning of the model using
the ResNet152 architecture. Fine-tuning pre-trained models like ResNet152 with advanced
frameworks such as Optuna helped optimize their adaptability to diverse datasets, thereby
improving their performance. The study also highlighted the importance of using robust
preprocessing techniques to enhance image quality and ensure accurate feature extraction.
The results demonstrate the potential of deep learning models, combined with hyperparameter
optimization techniques, to significantly improve the early detection of breast cancer, contributing
to more accurate, automated diagnoses. Moreover, this research provides valuable
insights into the application of AI in medical imaging, showing how optimization algorithms
can be leveraged to tackle complex, high-dimensional problems in healthcare.
The future perspectives include integrating the system into real clinical environments, enabling
faster and more accurate breast cancer detection, validated through large-scale clinical
studies. Improving the model’s generalization could be achieved by using more diverse
datasets, including images from different devices or populations. Continued optimization of
the model architecture, exploration of multi-modal learning with various imaging techniques,
and further hyperparameter optimization using new approaches are also promising avenues.
Additionally, enhancing the model’s interpretability for greater clinical confidence and applying
these techniques to other types of cancers could expand the impact of this research in
the field. Exploring the use of other models, such as EfficientNet or YOLO Detector, could
also offer potential improvements in accuracy and efficiency for breast cancer diagnosis.
