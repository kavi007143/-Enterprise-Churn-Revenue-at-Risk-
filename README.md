# 📊 B2B SaaS Enterprise Churn Engine & Revenue-at-Risk Framework

A production-grade Machine Learning solution designed to predict high-ticket corporate account churn, quantify millions of dollars in subscription revenue risk, and surface enterprise-level churn drivers using advanced Gradient Boosting architectures.

## 💰 Business Impact & Key Metrics
* **Identified \$9,427,464.00 in Annual Recurring Revenue (ARR)** across 42 high-ticket corporate accounts vulnerable to subscription contraction/churn.
* Successfully optimized machine learning architectures to **boost churn Recall score from 18% to 59%**, capturing the majority of revenue loss before it happened.
* Surfaced key operational bottlenecks, isolating that **EdTech sector clients (6.08%)** and **Customer Support Resolution Delays (4.01%)** are the highest drivers of customer churn.

## 🛠️ Data Engineering & Relational Pipeline
The project utilizes a complex relational enterprise database consisting of 5 connected tables:
1. `Accounts`: Primary corporate client metadata (500 records)
2. `Subscriptions`: Contract terms, MRR, ARR, and renewal flags
3. `Feature Usage Logs`: Granular user engagement logs (25,000+ rows)
4. `Support Tickets`: Helpdesk latency and satisfaction metrics
5. `Churn Events`: Cancellation timestamps and feedback logs

**Feature Engineering Pipeline:** Time-series feature logs and interaction datasets were aggregated by `account_id` using Pandas to engineering core attributes like `total_tickets`, `avg_resolution_time`, and contract values (`total_arr`, `total_mrr`) into a single flattened machine learning master table.

## ⚡ Tech Stack & Machine Learning Pipeline
* **Languages & Core Libraries:** Python, Pandas, NumPy, Scikit-Learn
* **Algorithms Deployed:** Random Forest Classifier, XGBoost Classifier
* **Imbalance Handling:** Customized class weight scaling (`scale_pos_weight=3.54` tuned to `6.0`) to account for severe 80:20 minority class imbalance.
* **Evaluation Focus:** Evaluated using **Recall and ROC-AUC (0.63)** instead of simple accuracy to prevent major corporate revenue leakages.

## 📈 Top Churn Drivers (Feature Importance)
1. **Industry (EdTech)** - 6.08%
2. **Industry (DevTools)** - 4.66%
3. **Total Seats Purchased** - 4.16%
4. **Average Support Resolution Time (Hours)** - 4.01%
5. **Total Monthly Recurring Revenue (MRR)** - 4.00%
