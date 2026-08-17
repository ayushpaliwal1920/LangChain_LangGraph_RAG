from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
class Student:
    def __init__(self,name):
        self.name = name

    def show(self):
        print(self.name)

def add(a,b):
    return a+b
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(python_code)

print(chunks)


## EXAMPLE 2 :

text = """
Title: Artificial Intelligence in Healthcare

Abstract

Artificial Intelligence (AI) is transforming the healthcare industry by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes. AI technologies such as machine learning, deep learning, and natural language processing are being integrated into various healthcare applications. This paper discusses the role of AI in healthcare, its benefits, challenges, and future prospects.

Introduction

Healthcare systems worldwide generate enormous amounts of data every day. Electronic health records, medical imaging, laboratory reports, and wearable devices contribute to this growing volume of information. Traditional methods of data analysis often struggle to process such large datasets efficiently.

Artificial Intelligence provides advanced techniques to analyze complex healthcare data. AI systems can identify patterns, predict diseases, and assist medical professionals in decision-making. The increasing adoption of AI has led to significant improvements in healthcare efficiency and quality.

Literature Review

Several studies have demonstrated the effectiveness of AI in medical diagnosis. Deep learning algorithms have achieved remarkable accuracy in image classification tasks such as detecting tumors in MRI scans and identifying diseases in X-ray images.

Natural Language Processing techniques have also been applied to clinical notes and patient records. Researchers have used NLP to extract valuable information from unstructured medical documents, improving clinical workflows and patient care.

Machine learning models have shown promising results in predicting disease outbreaks, patient readmissions, and treatment responses. These studies highlight the growing importance of AI-driven healthcare solutions.

Methodology

This study examines various AI applications in healthcare through a comprehensive review of existing literature and industry reports.

The methodology consists of three stages:

1. Data Collection
Healthcare-related research papers, case studies, and technical reports were collected from academic databases.

2. Data Analysis
The collected materials were analyzed to identify common trends, challenges, and opportunities associated with AI implementation.

3. Evaluation
Different AI techniques were evaluated based on their performance, scalability, and practical applicability in healthcare environments.

Results

The analysis revealed several significant findings.

First, AI-assisted diagnostic systems demonstrated higher accuracy compared to traditional rule-based systems. Deep learning models achieved particularly strong performance in medical imaging tasks.

Second, predictive analytics enabled hospitals to optimize resource allocation and improve patient management. Healthcare providers reported reduced operational costs and enhanced efficiency.

Third, NLP-based systems successfully automated documentation tasks, reducing administrative burden for healthcare professionals.

However, challenges such as data privacy concerns, model interpretability, and regulatory compliance continue to hinder widespread adoption.

Discussion

The results indicate that AI has substantial potential to improve healthcare delivery. Automated diagnostic tools can support clinicians in making informed decisions, while predictive models can enhance preventive care strategies.

Despite these advantages, ethical considerations remain critical. Healthcare organizations must ensure patient data security and transparency in AI decision-making processes. Furthermore, healthcare professionals require adequate training to effectively use AI-powered tools.

Future research should focus on developing explainable AI systems that provide clear reasoning behind their predictions. Such advancements could increase trust and facilitate broader adoption.

Conclusion

Artificial Intelligence is reshaping the healthcare industry through improved diagnostics, predictive analytics, and automation. While challenges related to privacy, ethics, and regulation remain, the benefits of AI are significant.

Continued research and collaboration between healthcare providers, policymakers, and technology companies will be essential for maximizing the potential of AI in healthcare. As AI technologies continue to evolve, they are expected to play an increasingly important role in improving global healthcare outcomes.

References

1. Smith, J. Artificial Intelligence in Modern Healthcare. Journal of Medical Informatics, 2023.

2. Brown, A. Deep Learning Applications in Radiology. Healthcare Technology Review, 2024.

3. Wilson, P. Natural Language Processing for Clinical Documentation. Medical AI Conference Proceedings, 2022."""

splitter = RecursiveCharacterTextSplitter(
    Language = Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])
