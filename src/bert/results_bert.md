To test the SOTA, we also implemented the BERT model with different tweaks that allow for better results. 

In order to evaluate the current State-of-the-Art (SOTA), we implemented a BERT-based architecture. Our approach moved beyond simple text classification by fine-tuning the model through a series of refinements designed to maximize contextual understanding.

# Experiment 1 - statement

This experiment established a performance baseline using only the raw text of the statements.

The model achieved a peak accuracy of 28.41%. While low, this significantly exceeds random chance (16.6% for 6 classes), proving BERT captures some lexical patterns of misinformation.
The highest precision (0.45) was found in the pants-fire category, suggesting that extreme lies use a distinct, sensationalist style that BERT identifies easily.
The barely-true class recorded the lowest F1-score (0.21). This highlights the model's struggle to distinguish subtle nuances of "partial truth" without external context.

These results confirm that the "statement-only" approach is insufficient. In the political domain, truthfulness is deeply tied to context, necessitating the inclusion of metadata (speaker, party, and subject) for better predictive power.

## Test 1
Epoch	Training Loss	Validation Loss	Accuracy
1	1.735347	1.691837	0.269470
2	1.636324	1.681138	0.267134
3	1.517180	1.717055	0.262461

Final results for Test Set: {
  'eval_loss': 1.7175817489624023, 
  'eval_accuracy': 0.2691397000789266, 
  'eval_runtime': 10.8867, 
  'eval_samples_per_second': 116.38, 
  'eval_steps_per_second': 7.348, 
  'epoch': 3.0
}

## Test 2
Epoch	Training Loss	Validation Loss	Accuracy
1	1.746786	1.704752	0.259346
2	1.676083	1.687270	0.267913
3	1.583930	1.714602	0.271807

Final results for Test Set: {
  'eval_loss': 1.673477053642273, 
  'eval_accuracy': 0.2841357537490134, 
  'eval_runtime': 10.927, 
  'eval_samples_per_second': 115.952, 
  'eval_steps_per_second': 7.321, 
  'epoch': 3.0
}

              precision    recall  f1-score   support

  pants-fire       0.45      0.26      0.33        92
       false       0.30      0.29      0.30       249
 barely-true       0.28      0.17      0.21       212
   half-true       0.26      0.34      0.30       265
 mostly-true       0.27      0.29      0.28       241
        true       0.28      0.33      0.30       208

    accuracy                           0.28      1267
   macro avg       0.31      0.28      0.29      1267
weighted avg       0.29      0.28      0.28      1267

# Experiment 2 - subject, speaker, party_affiliation, context, statement

## Test 1 - max_length = 128

In this stage, we integrated contextual metadata—Subject, Speaker, Party, and Venue—concatenated.

Adding context increased accuracy to 29.91% (a ~1.5% gain over the baseline). This demonstrates that BERT effectively utilizes source credibility and speaker history to refine its predictions.
While the model struggle with the 6-way granular classification, it shows strength in binary polarity. Misclassifications typically stay within the "Fake" cluster (pants-fire, false, barely-true) or the "Real" cluster (mostly-true, true).
The barely-true class remains the most difficult to predict, with the lowest recall (0.17). It is frequently confused with false or half-true, confirming that "partially false" statements lack unique linguistic markers even when context is provided.

Epoch	Training Loss	Validation Loss	Accuracy
1	1.746525	1.671923	0.291277
2	1.649708	1.634455	0.307632
3	1.552494	1.670749	0.303738

Final results for Test Set: {
  'eval_loss': 1.6560808420181274, 
  'eval_accuracy': 0.2991318074191002, 
  'eval_runtime': 10.4873, 
  'eval_samples_per_second': 120.813, 
  'eval_steps_per_second': 7.628, 
  'epoch': 3.0
}

              precision    recall  f1-score   support

  pants-fire       0.33      0.25      0.28        92
       false       0.32      0.36      0.34       249
 barely-true       0.30      0.17      0.21       212
   half-true       0.27      0.35      0.30       265
 mostly-true       0.31      0.31      0.31       241
        true       0.30      0.31      0.30       208

    accuracy                           0.30      1267
   macro avg       0.30      0.29      0.29      1267
weighted avg       0.30      0.30      0.30      1267

## Test 2 - max_length = 256, 5 epochs, LIME for visualisation

The decision to increase the max_length to 256 was crucial for accommodating the concatenated metadata, ensuring no critical contextual information was truncated. Through the application of LIME, we gained essential insights into the model's 'black-box' decision-making process. The analysis revealed that metadata fields, specifically 'Speaker' and 'Party', frequently exerted a higher influence on the final prediction than the 'Statement' itself.

While this confirms that the model successfully integrates context, it also uncovers a reliance on historical biases—where specific political figures or affiliations are statistically linked to lower truthfulness ratings in the dataset. This Explainable AI (XAI) approach was instrumental in verifying that our model transcends simple keyword matching, though it also highlights the challenge of isolating purely linguistic deception from source-based reputation. [img/exp2_test2_lime.png]

Epoch	Training Loss	Validation Loss	Accuracy
1	1.750509	1.671393	0.296729
2	1.651286	1.634846	0.297508
3	1.536917	1.719723	0.278037
4	1.193853	1.821102	0.286604
5	1.028424	1.919229	0.284268

Final results for Test Set: {
  'eval_loss': 1.9267863035202026, 
  'eval_accuracy': 0.27150749802683505, 
  'eval_runtime': 21.1448, 
  'eval_samples_per_second': 59.92, 
  'eval_steps_per_second': 3.783, 
  'epoch': 5.0
}

              precision    recall  f1-score   support

  pants-fire       0.29      0.27      0.28        92
       false       0.35      0.32      0.33       249
 barely-true       0.24      0.18      0.21       212
   half-true       0.26      0.33      0.29       265
 mostly-true       0.24      0.23      0.23       241
        true       0.27      0.28      0.27       208

    accuracy                           0.27      1267
   macro avg       0.27      0.27      0.27      1267
weighted avg       0.27      0.27      0.27      1267

## Test 3 - max_length = 256, [SEP] keyword

In this iteration, we extended the training to 5 epochs and maintained the [SEP] token structure. However, the results indicate a clear case of overfitting: while the training loss decreased significantly, the validation loss began to rise after the second epoch, leading to a drop in test accuracy to 27.38%.

Interestingly, this shift affected the model's 'internal logic,' as observed through LIME. For the same analyzed instance, the model's confidence shifted from half-true (0.35) to false (0.35). This suggests that as the model overfits, it becomes more sensitive to specific metadata markers (like 'Republican' or 'Democrat') rather than the semantic content of the statement. The increase in precision for the pants-fire class (0.41) further confirms that the model is becoming more aggressive in labeling extreme misinformation, even at the cost of overall accuracy.

Epoch	Training Loss	Validation Loss	Accuracy
1	1.739963	1.665547	0.286604
2	1.637931	1.638091	0.303738
3	1.509213	1.723436	0.296729
4	1.157024	1.869026	0.288941
5	0.990860	1.993613	0.289720

Final results for Test Set: {
  'eval_loss': 1.9794371128082275, 
  'eval_accuracy': 0.27387529597474347, 
  'eval_runtime': 21.1236, 
  'eval_samples_per_second': 59.98, 
  'eval_steps_per_second': 3.787, 
  'epoch': 5.0
}

              precision    recall  f1-score   support

  pants-fire       0.41      0.28      0.33        92
       false       0.29      0.35      0.32       249
 barely-true       0.25      0.23      0.24       212
   half-true       0.26      0.29      0.27       265
 mostly-true       0.27      0.25      0.26       241
        true       0.26      0.23      0.24       208

    accuracy                           0.27      1267
   macro avg       0.29      0.27      0.28      1267
weighted avg       0.28      0.27      0.27      1267

# Experiment 3 - RoBERTa

Epoch	Training Loss	Validation Loss	Accuracy
1	1.671382	1.640174	0.304517
2	1.618599	1.628159	0.308411
3	1.501099	1.638132	0.301402

Final results for Test Set: {
  'eval_loss': 1.646820068359375, 
  'eval_accuracy': 0.27940015785319655, 
  'eval_runtime': 20.6492, 
  'eval_samples_per_second': 61.358, 
  'eval_steps_per_second': 7.7, 
  'epoch': 3.0
}

              precision    recall  f1-score   support

  pants-fire       0.43      0.22      0.29        92
       false       0.29      0.33      0.31       249
 barely-true       0.25      0.14      0.18       212
   half-true       0.26      0.31      0.28       265
 mostly-true       0.29      0.32      0.30       241
        true       0.27      0.31      0.29       208

    accuracy                           0.28      1267
   macro avg       0.30      0.27      0.27      1267
weighted avg       0.28      0.28      0.28      1267

## Binary Classes

Epoch	Training Loss	Validation Loss	Accuracy
1	1.270325	1.299706	0.647975
2	1.218584	1.256212	0.654984
3	1.138134	1.264601	0.658100

Final results for Test Set: {
  'eval_loss': 1.2998323440551758, 
  'eval_accuracy': 0.6550907655880032, 
  'eval_runtime': 12.7846, 
  'eval_samples_per_second': 99.104, 
  'eval_steps_per_second': 6.258, 
  'epoch': 3.0
}

              precision    recall  f1-score   support

        Fake       0.64      0.48      0.55       553
        Real       0.66      0.79      0.72       714

    accuracy                           0.66      1267
   macro avg       0.65      0.64      0.64      1267
weighted avg       0.65      0.66      0.65      1267