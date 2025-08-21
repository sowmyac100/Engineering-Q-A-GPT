import PyPDF2
from transformers import pipeline
import spacy

# Load pre-trained BERT model for Q&A
qa_pipeline = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-finetuned-squad")

# Load spaCy for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Extract named entities (key concepts like spacecraft components, systems, etc.)
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'MATERIAL']]
    return entities

# Generate Q&A pairs from a text section
def generate_qa_from_text(text, questions):
    qa_pairs = []
    for question in questions:
        answer = qa_pipeline(question=question, context=text)
        qa_pairs.append((question, answer['answer']))
    return qa_pairs

# Generate dynamic questions based on identified entities
def generate_dynamic_questions(entities):
    questions = []
    for entity in entities:
        questions.append(f"What is the role of {entity}?")
        questions.append(f"How does {entity} work?")
        questions.append(f"Why is {entity} important?")
    return questions

# Process multiple PDFs and generate Q&A pairs
def process_pdfs(pdf_paths):
    all_qa_pairs = []
    
    for pdf_path in pdf_paths:
        print(f"Processing PDF: {pdf_path}")
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Extract entities from the text
        entities = extract_entities(pdf_text)
        
        # Generate dynamic questions based on the extracted entities
        dynamic_questions = generate_dynamic_questions(entities)
        
        # Generate Q&A pairs for the current PDF
        qa_pairs = generate_qa_from_text(pdf_text, dynamic_questions)
        
        # Store the Q&A pairs
        all_qa_pairs.extend(qa_pairs)
        
    return all_qa_pairs

# Example usage
pdf_files = ["spacecraft_manual_1.pdf", "spacecraft_manual_2.pdf"]  # List your PDF files here

# Process all PDFs and generate Q&A pairs
qa_pairs = process_pdfs(pdf_files)

# Display the generated Q&A pairs
for question, answer in qa_pairs:
    print(f"Q: {question}")
    print(f"A: {answer}\n")
