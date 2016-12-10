# DistributedLearningPredictor


## Introduction
Electronic Health Record (EHR) data from multiple hospitals are usually hard to transfer or store at one place for privacy concerns or regulatory hurdles.  Following the work of Farhan et al. (2016), we develop three models which can learn from multiple databases distributedly and at the same time make predictions for patient diagnoses.  

## Sample data
Our package provides a small sample data including a training data set and a testing data set.  To mimic the situation when the data from two hospitals cannot be obtained at the same time, we split the training data to two parts as our example data.  We purposely choose a small dataset as the second part data to shorten the computational time using dropout update.  

## Real data preparation

We inherit the data preprocessing steps from https://github.com/wael34218/SequentialPhenotypePredictor.  It is highly recommended to install MIMICIII database and follow the data preparation steps in this repository.  The data preparation codes are also included in our package.  After the preparation steps, EHR data for one single patient looks like:

> d_401,d_486|{"black": 0, "hispanic": 0, "other": 0, "age": 69.90554414784394, "mideast": 0, "multi": 0, "gender": 0, "hawaiian": 0, "portuguese": 0, "american": 0, "asian": 0, "white": 1}|p_ASA81 p_CALCG2gi100NS p_CALCG2100NS p_CALCG2100NS p_HEPA5I l_50970 l_51265 p_ACET325 p_HYDR2 p_VANC1F p_VANCOBASE p_HEPA10SYR p_HEPA10SYR p_METO25 l_50862 l_50954 p_POTA20 l_50924 l_50953 l_50998 d_038 d_285.9 d_401 d_486 d_584 d_995|p_ASA325 p_D545NS1000 p_DEX50SY p_DOCU100 p_DOCU100L l_51214 d_401 d_486

For more detail, please refer to https://github.com/wael34218/SequentialPhenotypePredictor.

## Prediction with DNCE

To run Distributed Noise-Contrastive Estimation model (DNCE) run the following commands: 

Go to directory
   
    cd ../../Prediction

Create vocabulary for the first dataset

    python predictor.py --window 30 --size 350 --decay 8 --task CreateVocab --inputfilename train_p1_data --outputfilename1 raw_vocab_example1.pkl

Create vocabulary for the second dataset

    python predictor.py --window 30 --size 350 --decay 8 --task CreateVocab --inputfilename train_p2_data --outputfilename1 raw_vocab_example2.pkl

Merge two vocabularies and initialize an empty global model

    python predictor.py --window 30 --size 350 --decay 8 --task MergeVocab --outputfilename1 raw_vocab_example1.pkl --outputfilename2 raw_vocab_example2.pkl --savemodelname initglobalmodel.pkl
    
Another version to merge two vocabularies: add Differential Privacy (DP)

    python predictor.py --window 30 --size 350 --decay 8 --task MergeVocab --outputfilename1 raw_vocab_example1.pkl --outputfilename2 raw_vocab_example2.pkl --savemodelname initglobalmodel.pkl --privacyprotect 1 --clusternumber 6

Train the global model with the first dataset

    python predictor.py --window 30 --size 350 --decay 8 --task DNCEInit --savemodelname initglobalmodel.pkl --inputfilename train_p1_data --trainedmodelname trainedwithD1.pkl

Train the global model with the second dataset

    python predictor.py --window 30 --size 350 --decay 8 --task DNCEUpdate --savemodelname trainedwithD1.pkl --inputfilename train_p2_data --trainedmodelname trainedwithD1D2.pkl

## Prediction with Naive updates

To run Naive update model, run the following commands:

Initialize and train model with the first dataset

    python predictor.py --window 30 --size 350 --decay 8 --task InitTrain --inputfilename train_p1_data --trainedmodelname NVtrainedwithD1.pkl

Naive update the model with the second dataset

    python predictor.py --window 30 --size 350 --decay 8 --task NaiveUpdate --savemodelname NVtrainedwithD1.pkl --inputfilename train_p2_data --trainedmodelname NVtrainedwithD1D2.pkl

## Prediction with Dropout updates

To run Dropout update model, run the following commands:

Initialize and train model with the first dataset

    python predictor.py --window 30 --size 350 --decay 8 --task InitTrain --inputfilename train_p1_data --trainedmodelname DOtrainedwithD1.pkl

Dropout update the model with the second dataset

    python predictor.py --window 30 --size 350 --decay 8 --task DropoutUpdate --savemodelname DOtrainedwithD1.pkl --inputfilename train_p2_data --trainedmodelname DOtrainedwithD1D2.pkl

## Generate gold standard model

To run a gold standard model with all the training data, run the following commands:

    python predictor.py --window 30 --size 350 --decay 8 --task InitTrain --inputfilename train_all_data --trainedmodelname goldstandard_1.pkl

## Calculate PTK and AUC

To calculate the PTK of a target model versus the gold standard model:

    python predictor.py --window 30 --size 350 --decay 8 --task CalcPTK --savemodelname trainedwithD1D2.pkl --goldmodelname goldstandard_1.pkl


To calculate the stats of a target model including AUC:

    python predictor.py --window 30 --size 350 --decay 8 --task CalcStats --savemodelname trainedwithD1D2.pkl --inputfilename test_data
    
Arguments of predictor.py:
+  --window *int* the maximum distance between the current and predicted word within a sentence.
+  --size *int* the dimension of vector representations.
+  --decay *int* decay parameter in PDPS model.  Larger decay means less impact of a far-away event.
+  --task *str* 
                  1. "CreateVocab": create vocabulary from an input data.
                  2. "MergeVocab": merge two vocabularies.
                  3. "DNCEInit": DNCE initial train.
                  4. "DNCEUpdate": DNCE update train.
                  5. "InitTrain": initial train for naive updates or dropout udpates.  Or calculate gold standard model.
                  6. “NaiveUpdate”: Naive update train.
                  7. "DropoutUpdate": Dropout update train.
                  8. "CalcPTK": Calculate the PTK of one target model versus gold standard model.
                  9. "CalcStats": Calculate the stats including AUC. 
+  --inputfilename *str* the name of a input data file.
+  --outputfilename1 *str* the name of an output/input data file.
+  --outputfilename2 *str* the name of another output/input data file.
+  --savemodelname *str* the name of a saved model.
+  --trainedmodelname *str* the name of a trained model.
+  --goldmodelname *str* the name of a gold standard model.
+  --privacyprotect *int* whether add differential privacy in vocabulary merge process.
+  --clusternumber *int* if add privacy protection, how many clusters should be used.
  
## Libraries Used

This project depends on:

1. word2vec - https://code.google.com/p/word2vec/
2. ICD9 - https://github.com/sirrice/icd9
3. ICD9 - https://github.com/kshedden/icd9
4. gensim - http://radimrehurek.com/gensim/

## Sample data reference:
MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available from: http://www.nature.com/articles/sdata201635
