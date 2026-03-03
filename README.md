

---

**NovaBurn AI | Deep Learning + XAI for Burn Severity Classification** 🏆 *BSc Dissertation (89% Accuracy)*

**The Challenge:** Burn injuries cause ~180,000 deaths annually, with 89% occurring in low/middle-income countries lacking specialist care. Manual visual inspection—the current standard—is subjective (κ=0.52-0.67), time-consuming, and inconsistent, leading to misdiagnosis and delayed treatment.

**The Solution:** Developed a hybrid deep learning ensemble combining VGG19, InceptionV3, and ResNet50 with Explainable AI (XAI) to classify burn severity (1st, 2nd, 3rd degree) with **89.07% accuracy**, outperforming individual models (81-85%).

**Key Technical Achievements:**

• **Ensemble Architecture:** Weighted voting system (VGG19:0.4, Inception:0.35, ResNet:0.25) achieving 96.8% ROC-AUC with macro precision 0.92, recall 0.88, F1-score 0.90
• **Explainable AI Integration:** Dual XAI framework combining Grad-CAM heatmaps (87% correlation with expert annotations) + LIME pixel-level attribution for clinical transparency
• **Clinical Decision Support:** Automated PDF reports + AI chatbot providing evidence-based treatment protocols (ABA/WHO guidelines)
• **Deployment Ready:** Progressive Web App with <2min inference, offline capability, and batch processing for mass casualty events

**Technical Stack:** Python, TensorFlow/Keras, Flask, HTML/CSS, Grad-CAM, LIME, Google Colab (GPU training), Git/GitHub

**Validation:** Comprehensive testing (60+ test cases) with benchmarking against existing solutions (Suha & Sanam 2022: 96.42%, Jacobson 2023: 88%, Karthik 2021: 80.02%). Validated on Kaggle + GitHub datasets with extensive data augmentation.

**Impact:** Addresses critical healthcare disparity by democratizing specialist-level burn assessment—potentially reducing diagnostic time by 30.6% and saving an estimated $2.4M annually in Sri Lanka's healthcare system.


---

