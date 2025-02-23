# AI-Powered Banking Fraud Detection  

## Overview  
This project was developed for the **"2024-2025 IMI Big Data & Artificial Intelligence Competition UTM"**, hosted at **University of Toronto** with data provided by **Scotiabank**.  
It applies AI techniques to detect fraudulent banking transactions, including:  
- Unsupervised learning models for anomaly detection  
- Customer embeddings for advanced fraud detection  
- Visualization of anomalies using PCA  

## Installation & Setup

Since setup is quite simple, I did not use Docker for this project.
Please follow the instructions below to run the project on your local machine.

### **1. Install Dependencies**  
Ensure you are using **Python 3.10**. Then, install the required dependencies:  
```bash
pip install -r requirements.txt
```

### **2. Run the Main Script**  
Execute the following command to process data and visualize fraud detection results:  
```bash
python main.py
```

### **3. View Results**
After running the script, visualization will pop up.
Once you close the window, the program will output the results to the console.
```
Epoch 50/50
508/508 [==============================] - 0s 598us/step - loss: 0.0170
508/508 [==============================] - 0s 281us/step
2025-02-23 00:27:30.718 python[78341:5203633] +[IMKClient subclass]: chose IMKClient_Modern
2025-02-23 00:27:30.718 python[78341:5203633] +[IMKInputSession subclass]: chose IMKInputSession_Modern
detected fraudulent customers:
            customer_id  total_amount  transaction_count  anomaly_score
87     SYNCID0000000092  1.097015e+05               15.0             -1
367    SYNCID0000000386  2.289914e+07              217.0             -1
408    SYNCID0000000431  1.620077e+07             1475.0             -1
445    SYNCID0000000470  2.560108e+05              208.0             -1
466    SYNCID0000000491  1.114758e+07             1441.0             -1
...                 ...           ...                ...            ...
15821  SYNCID0000016728  2.348067e+07              290.0             -1
16040  SYNCID0000016958  1.205952e+07             1618.0             -1
16071  SYNCID0000016991  1.353024e+08             1135.0             -1
16170  SYNCID0000017095  1.607286e+07              185.0             -1
16209  SYNCID0000017137  4.775002e+07              500.0             -1

[163 rows x 4 columns]
Total number of fraudulent customers: 163
Total number of customers: 16255
Percentage of fraudulent customers: 1.00276837896032
```

## **Credits**  
- **Author:** Yehyun Lee – I worked alone, and all code is written by me.  
- **Acknowledgments:**  
  - **Hisham** helped early in the project with brainstorming.  
  - **Tyseer** joined the Discord server but did not participate.  
  - **Jasrita** did not communicate at all.  

---

Enjoy detecting frauds! 🚀 \
-- Yehyun Lee
