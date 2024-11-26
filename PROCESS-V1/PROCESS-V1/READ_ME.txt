Welcome to the PROCESS challenge 

There are 157 subjects across the train and development sets provided here. Each folder has the name Process-rec-XXX, where XXX is the Record-ID. 

Each folder contains both the audio and the manual transcriptions for the three tasks: CTD (Cookie Theft description), PFT (phonemic fluency test) and SFT (semantic fluency test). For example, the audio and transcription files are Process-rec-XXX__CTD.wav and Process-rec-XXX__CTD.txt inside each folder for the CTD task. 

Please refer to https://processchallenge.github.io/dataset/ for a detailed dataset description. 

The "dem-info.csv" file contains the labels and metadata for all the subjects. The columns in this CSV file are: 
"Record-ID": the subject identifiers that are the folder names
"TrainOrDev": if the subject belong to the train or development (dev) set
"Class": three classes represented by the subjects that are either Dementia, MCI or HC 
"Gender": if the subject is male or female
"Age": age of the subject while participating in the study. For participants without a known age, we have provided the average age across the corpus and marked this with an asterisk (66*)  
"Converted-MMSE": Mini-Mental State Exam (MMSE) score for the subjects, if available, otherwise the MMSE value has been converted from Moca (Fasnacht et al, 2023) and ACE-III (María-Guiu, 2018).

The instructions for classification and regression tasks can be found at: https://processchallenge.github.io/instructions/

All the best,
The PROCESS challenge organisers

---

REFERENCES:
Moca2MMSE
https://agsjournals.onlinelibrary.wiley.com/doi/full/10.1111/jgs.18124
Fasnacht, J.S., Wueest, A.S., Berres, M., Thomann, A.E., Krumm, S., Gutbrod, K., Steiner, L.A., Goettel, N. and Monsch, A.U., 2023. Conversion between the Montreal cognitive assessment and the mini‐mental status examination. Journal of the American Geriatrics Society, 71(3), pp.869-879.


Ace2MMSE
https://www.cambridge.org/core/journals/international-psychogeriatrics/article/abs/conversion-between-addenbrookes-cognitive-examination-iii-and-minimental-state-examination/4FCFF17328AAE98AE2AC5307F769DEE1
Matías-Guiu, J.A., Pytel, V., Cortés-Martínez, A., Valles-Salgado, M., Rognoni, T., Moreno-Ramos, T. and Matías-Guiu, J., 2018. Conversion between Addenbrooke's cognitive examination III and mini-mental state examination. International 
